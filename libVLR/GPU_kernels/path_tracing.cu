#include "cameras.cu"
#include "light_transport_common.cuh"

namespace VLR {
    // Context-scope Variables
    rtDeclareVariable(optix::uint2, pv_imageSize, , );
    rtDeclareVariable(uint32_t, pv_numAccumFrames, , );
    rtDeclareVariable(ProgSigSampleLensPosition, pv_progSampleLensPosition, , );
    rtDeclareVariable(ProgSigSampleIDF, pv_progSampleIDF, , );
    rtBuffer<KernelRNG, 2> pv_rngBuffer;
    rtBuffer<SpectrumStorage, 2> pv_outputBuffer;
    //rtBuffer<RGBSpectrum, 2> pv_outputNormalBuffer;
    //rtBuffer<RGBSpectrum, 2> pv_outputAlbedoBuffer;

	

	#define CLAMPINTENSITY(c, clampValue)		const float v=std::fmax(c.r,std::fmax(c.g, c.b)); \
							if(v>clampValue){const float m=clampValue/v;c.r*=m; \
							c.g*=m;c.b*=m; /* don't touch w */ }

	
    // Common Closest Hit Program for All Primitive Types and Materials
    RT_PROGRAM void pathTracingIteration() {
		// first hit is an object
        if (sm_payload.contribution.a == -1.0f)
            sm_payload.contribution.a = 1.f;

        KernelRNG &rng = sm_payload.rng;
        WavelengthSamples &wls = sm_payload.wls;

        SurfacePoint surfPt;
        float hypAreaPDF;
        calcSurfacePoint(&surfPt, &hypAreaPDF);

        const SurfaceMaterialDescriptor matDesc = pv_materialDescriptorBuffer[pv_materialIndex];
        BSDF bsdf(matDesc, surfPt, wls);
        EDF edf(matDesc, surfPt, wls);

        Vector3D dirOutLocal = surfPt.shadingFrame.toLocal(-asVector3D(sm_ray.direction));

        // implicit light sampling
        SampledSpectrum spEmittance = edf.evaluateEmittance();
        if (spEmittance.hasNonZero()) {
            SampledSpectrum Le = spEmittance * edf.evaluate(EDFQuery(), dirOutLocal);

            float MISWeight = 1.0f;
            if (!sm_payload.prevSampledType.isDelta() && sm_ray.ray_type != RayType::Primary) {
                float bsdfPDF = sm_payload.prevDirPDF;
                float dist2 = surfPt.calcSquaredDistance(asPoint3D(sm_ray.origin));
                float lightPDF = pv_importance / getSumLightImportances() * hypAreaPDF * dist2 / std::fabs(dirOutLocal.z);
                MISWeight = (bsdfPDF * bsdfPDF) / (lightPDF * lightPDF + bsdfPDF * bsdfPDF);
            }

            sm_payload.contribution += sm_payload.alpha * Le * MISWeight;
			CLAMPINTENSITY(sm_payload.contribution, 10.f);
        }
        if (surfPt.atInfinity || sm_payload.maxLengthTerminate)
            return;

        // Russian roulette
        float continueProb = std::fmin(sm_payload.alpha.importance(wls.selectedLambdaIndex()) / sm_payload.initImportance, 1.0f);
        if (rng.getFloat0cTo1o() >= continueProb)
            return;
        sm_payload.alpha /= continueProb;

        Normal3D geomNormalLocal = surfPt.shadingFrame.toLocal(surfPt.geometricNormal);
        BSDFQuery fsQuery(dirOutLocal, geomNormalLocal, DirectionType::All(), wls);

        // get base color for denoiser
   /*     if (sm_payload.contribution.a == -1.0f) {
            sm_payload.contribution.a = 1.f;

      /*      const BSDFProcedureSet procSet = pv_bsdfProcedureSetBuffer[matDesc.bsdfProcedureSetIndex];
            auto progGetBaseColor = (ProgSigBSDFGetBaseColor)procSet.progGetBaseColor;
            sm_payload.albedo = progGetBaseColor((const uint32_t *)&bsdf);
            //	sm_payload.normal = RGBSpectrum(surfPt.geometricNormal.x, surfPt.geometricNormal.y, surfPt.geometricNormal.z);
            //	sm_payload.normal = RGBSpectrum(geomNormalLocal.x, geomNormalLocal.y, geomNormalLocal.z);
    /*		auto rotMat = Matrix4x4(pv_perspectiveCamera.orientation.toMatrix3x3());
            //rotMat = rotateY(1.57f) * rotMat;
            rotMat = translate(pv_perspectiveCamera.position.x, pv_perspectiveCamera.position.y, pv_perspectiveCamera.position.z) * rotMat;
            rotMat = invert(rotMat);
            auto normalCam = normalize(rotMat * surfPt.geometricNormal);
            sm_payload.normal = RGBSpectrum(-normalCam.x, normalCam.y, -normalCam.z);
    *///	}

        // Next Event Estimation (explicit light sampling)
        if (bsdf.hasNonDelta()) {
            SurfaceLight light;
            float lightProb;
            float uPrim;
            selectSurfaceLight(rng.getFloat0cTo1o(), &light, &lightProb, &uPrim);

            SurfaceLightPosSample lpSample(uPrim, rng.getFloat0cTo1o(), rng.getFloat0cTo1o());
            SurfaceLightPosQueryResult lpResult;
            light.sample(lpSample, &lpResult);

            const SurfaceMaterialDescriptor lightMatDesc = pv_materialDescriptorBuffer[lpResult.materialIndex];
            EDF ledf(lightMatDesc, lpResult.surfPt, wls);
            SampledSpectrum M = ledf.evaluateEmittance();

            Vector3D shadowRayDir;
            float squaredDistance;
            float fractionalVisibility;
			SampledSpectrum shadow_color;
            if (M.hasNonZero() && testVisibility(surfPt, lpResult.surfPt, &shadowRayDir, &squaredDistance, &fractionalVisibility, &shadow_color)) {
                Vector3D shadowRayDir_l = lpResult.surfPt.toLocal(-shadowRayDir);
                Vector3D shadowRayDir_sn = surfPt.toLocal(shadowRayDir);

				SampledSpectrum Le = M * ledf.evaluate(EDFQuery(), shadowRayDir_l) *shadow_color;
                float lightPDF = lightProb * lpResult.areaPDF;

                SampledSpectrum fs = bsdf.evaluate(fsQuery, shadowRayDir_sn);
                float cosLight = lpResult.surfPt.calcCosTerm(-shadowRayDir);
                float bsdfPDF = bsdf.evaluatePDF(fsQuery, shadowRayDir_sn) * cosLight / squaredDistance;

                float MISWeight = 1.0f;
                if (!lpResult.posType.isDelta() && !std::isinf(lightPDF))
                    MISWeight = (lightPDF * lightPDF) / (lightPDF * lightPDF + bsdfPDF * bsdfPDF);

                float G = fractionalVisibility * absDot(shadowRayDir_sn, geomNormalLocal) * cosLight / squaredDistance;
                float scalarCoeff = G * MISWeight / lightPDF; // 直接contributionの計算式に入れるとCUDAのバグなのかおかしな結果になる。
                sm_payload.contribution += sm_payload.alpha * Le * fs * scalarCoeff;
				CLAMPINTENSITY(sm_payload.contribution, 100.f);
            }
        }

        BSDFSample sample(rng.getFloat0cTo1o(), rng.getFloat0cTo1o(), rng.getFloat0cTo1o());
        BSDFQueryResult fsResult;
        SampledSpectrum fs = bsdf.sample(fsQuery, sample, &fsResult);
        if (fs == SampledSpectrum::Zero() || fsResult.dirPDF == 0.0f)
            return;
        if (fsResult.sampledType.isDispersive() && !wls.singleIsSelected()) {
            fsResult.dirPDF /= SampledSpectrum::NumComponents();
            wls.setSingleIsSelected();
        }

        float cosFactor = dot(fsResult.dirLocal, geomNormalLocal);
        sm_payload.alpha *= fs * (std::fabs(cosFactor) / fsResult.dirPDF);
	//	CLAMPINTENSITY(sm_payload.alpha);

        Vector3D dirIn = surfPt.fromLocal(fsResult.dirLocal);
        sm_payload.origin = offsetRayOrigin(surfPt.position, cosFactor > 0.0f ? surfPt.geometricNormal : -surfPt.geometricNormal);
        sm_payload.direction = dirIn;
        sm_payload.prevDirPDF = fsResult.dirPDF;
        sm_payload.prevSampledType = fsResult.sampledType;
        sm_payload.terminate = false;
    }



    // JP: 本当は無限大の球のIntersection/Bounding Box Programを使用して環境光に関する処理もClosest Hit Programで統一的に行いたい。
    //     が、OptiXのBVHビルダーがLBVHベースなので無限大のAABBを生成するのは危険。
    //     仕方なくMiss Programで環境光を処理する。
    RT_PROGRAM void pathTracingMiss() {
        // first hit is the background, set alpha to °
        if (sm_payload.contribution.a == -1.0f) {
       //     sm_payload.albedo = spEmittance;
            sm_payload.contribution.a = 0.f;
        }

        if (pv_envLightDescriptor.importance == 0)
            return;

        Vector3D direction = asVector3D(sm_ray.direction);
        float phi, theta;
        direction.toPolarYUp(&theta, &phi);

        float sinPhi, cosPhi;
        VLR::sincos(phi, &sinPhi, &cosPhi);
        Vector3D texCoord0Dir = normalize(Vector3D(-cosPhi, 0.0f, -sinPhi));
        ReferenceFrame shadingFrame;
        shadingFrame.x = texCoord0Dir;
        shadingFrame.z = -direction;
        shadingFrame.y = cross(shadingFrame.z, shadingFrame.x);

        SurfacePoint surfPt;
        surfPt.position = Point3D(direction.x, direction.y, direction.z);
        surfPt.shadingFrame = shadingFrame;
        surfPt.isPoint = false;
        surfPt.atInfinity = true;

        surfPt.geometricNormal = -direction;
        surfPt.u = phi;
        surfPt.v = theta;
        phi += pv_envLightDescriptor.body.asInfSphere.rotationPhi;
        phi = phi - std::floor(phi / (2 * M_PIf)) * 2 * M_PIf;
        surfPt.texCoord = TexCoord2D(phi / (2 * M_PIf), theta / M_PIf);

        float hypAreaPDF = evaluateEnvironmentAreaPDF(phi, theta);

        const SurfaceMaterialDescriptor matDesc = pv_materialDescriptorBuffer[pv_envLightDescriptor.materialIndex];
        EDF edf(matDesc, surfPt, sm_payload.wls);

        Vector3D dirOutLocal = surfPt.shadingFrame.toLocal(-asVector3D(sm_ray.direction));

        // implicit light sampling
        SampledSpectrum spEmittance = edf.evaluateEmittance();
        if (spEmittance.hasNonZero()) {
            SampledSpectrum Le = spEmittance * edf.evaluate(EDFQuery(), dirOutLocal);

            float MISWeight = 1.0f;
            if (!sm_payload.prevSampledType.isDelta() && sm_ray.ray_type != RayType::Primary) {
                float bsdfPDF = sm_payload.prevDirPDF;
                float dist2 = surfPt.calcSquaredDistance(asPoint3D(sm_ray.origin));
                float lightPDF = pv_envLightDescriptor.importance / getSumLightImportances() * hypAreaPDF * dist2 / std::fabs(dirOutLocal.z);
                MISWeight = (bsdfPDF * bsdfPDF) / (lightPDF * lightPDF + bsdfPDF * bsdfPDF);
            }

            sm_payload.contribution += sm_payload.alpha * Le * MISWeight;
			CLAMPINTENSITY(sm_payload.contribution, 100.f);
        }
    }


    // Common Ray Generation Program for All Camera Types
    RT_PROGRAM void pathTracing() {
        KernelRNG rng = pv_rngBuffer[sm_launchIndex];

        optix::float2 p = make_float2(sm_launchIndex.x + rng.getFloat0cTo1o(), sm_launchIndex.y + rng.getFloat0cTo1o());

        float selectWLPDF;
        WavelengthSamples wls = WavelengthSamples::createWithEqualOffsets(rng.getFloat0cTo1o(), rng.getFloat0cTo1o(), &selectWLPDF);

        LensPosSample We0Sample(rng.getFloat0cTo1o(), rng.getFloat0cTo1o());
        LensPosQueryResult We0Result;
        SampledSpectrum We0 = pv_progSampleLensPosition(wls, We0Sample, &We0Result);

        IDFSample We1Sample(p.x / pv_imageSize.x, p.y / pv_imageSize.y);
        IDFQueryResult We1Result;
        SampledSpectrum We1 = pv_progSampleIDF(We0Result.surfPt, wls, We1Sample, &We1Result);

        Vector3D rayDir = We0Result.surfPt.fromLocal(We1Result.dirLocal);
        SampledSpectrum alpha = (We0 * We1) * (We0Result.surfPt.calcCosTerm(rayDir) / (We0Result.areaPDF * We1Result.dirPDF * selectWLPDF));

        optix::Ray ray = optix::make_Ray(asOptiXType(We0Result.surfPt.position), asOptiXType(rayDir), RayType::Primary, 0.0f, FLT_MAX);

        Payload payload;
        payload.maxLengthTerminate = false;
        payload.rng = rng;
        payload.initImportance = alpha.importance(wls.selectedLambdaIndex());
        payload.wls = wls;
        payload.alpha = alpha;
        payload.contribution = SampledSpectrum::Zero();
		payload.contribution.a = -1.0f;
        //payload.normal = SampledSpectrum(0.0, 1.0, 0.0);
        //payload.albedo = SampledSpectrum(-1.f, -1.f, -1.f);

        const uint32_t MaxPathLength = 25;
        uint32_t pathLength = 0;
        while (true) {
            payload.terminate = true;
            ++pathLength;
            if (pathLength >= MaxPathLength)
                payload.maxLengthTerminate = true;
            rtTrace(pv_topGroup, ray, payload);

            if (payload.terminate)
                break;
            VLRAssert(pathLength < MaxPathLength, "Path should be terminated... Something went wrong...");

            ray = optix::make_Ray(asOptiXType(payload.origin), asOptiXType(payload.direction), RayType::Scattered, 0.0f, FLT_MAX);
        }
        pv_rngBuffer[sm_launchIndex] = payload.rng;
        if (!payload.contribution.allFinite()) {
           //	vlrprintf("Pass %u, (%u, %u): Not a finite value.\n", pv_numAccumFrames, sm_launchIndex.x, sm_launchIndex.y);
            return;
        }

        if (pv_numAccumFrames == 1) {
            pv_outputBuffer[sm_launchIndex].reset();
        //	pv_outputNormalBuffer[sm_launchIndex] = payload.normal;
        //	pv_outputAlbedoBuffer[sm_launchIndex] = payload.albedo;
        }
		//CLAMPINTENSITY(payload.contribution);
        pv_outputBuffer[sm_launchIndex].add(wls, payload.contribution);
    }

	

    // Exception Program
    RT_PROGRAM void exception() {
        //uint32_t code = rtGetExceptionCode();
        rtPrintExceptionDetails();
    }
}
