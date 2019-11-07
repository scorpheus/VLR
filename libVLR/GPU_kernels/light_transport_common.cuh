﻿#pragma once

#include "kernel_common.cuh"
#include <optix_math.h>

namespace VLR {
    // Context-scope Variables
    rtDeclareVariable(rtObject, pv_topGroup, , );

    rtDeclareVariable(DiscreteDistribution1D, pv_lightImpDist, , );
    rtDeclareVariable(GeometryInstanceDescriptor, pv_envLightDescriptor, , );



    class BSDF {
#define VLR_MAX_NUM_BSDF_PARAMETER_SLOTS (32)
        uint32_t data[VLR_MAX_NUM_BSDF_PARAMETER_SLOTS];

        //ProgSigBSDFGetBaseColor progGetBaseColor;
        ProgSigBSDFmatches progMatches;
        ProgSigBSDFSampleInternal progSampleInternal;
        ProgSigBSDFEvaluateInternal progEvaluateInternal;
        ProgSigBSDFEvaluatePDFInternal progEvaluatePDFInternal;

        RT_FUNCTION bool matches(DirectionType dirType) {
            return progMatches((const uint32_t*)this, dirType);
        }
        RT_FUNCTION SampledSpectrum sampleInternal(const BSDFQuery &query, float uComponent, const float uDir[2], BSDFQueryResult* result) {
            return progSampleInternal((const uint32_t*)this, query, uComponent, uDir, result);
        }
        RT_FUNCTION SampledSpectrum evaluateInternal(const BSDFQuery &query, const Vector3D &dirLocal) {
            return progEvaluateInternal((const uint32_t*)this, query, dirLocal);
        }
        RT_FUNCTION float evaluatePDFInternal(const BSDFQuery &query, const Vector3D &dirLocal) {
            return progEvaluatePDFInternal((const uint32_t*)this, query, dirLocal);
        }

    public:
        RT_FUNCTION BSDF(const SurfaceMaterialDescriptor &matDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls) {
            ProgSigSetupBSDF setupBSDF = (ProgSigSetupBSDF)matDesc.progSetupBSDF;
            setupBSDF(matDesc.data, surfPt, wls, (uint32_t*)this);

            const BSDFProcedureSet procSet = pv_bsdfProcedureSetBuffer[matDesc.bsdfProcedureSetIndex];

            //progGetBaseColor = (ProgSigBSDFGetBaseColor)procSet.progGetBaseColor;
            progMatches = (ProgSigBSDFmatches)procSet.progMatches;
            progSampleInternal = (ProgSigBSDFSampleInternal)procSet.progSampleInternal;
            progEvaluateInternal = (ProgSigBSDFEvaluateInternal)procSet.progEvaluateInternal;
            progEvaluatePDFInternal = (ProgSigBSDFEvaluatePDFInternal)procSet.progEvaluatePDFInternal;
        }

        //RT_FUNCTION SampledSpectrum getBaseColor() {
        //    return progGetBaseColor((const uint32_t*)this);
        //}

        RT_FUNCTION bool hasNonDelta() {
            return matches(DirectionType::WholeSphere() | DirectionType::NonDelta());
        }

        RT_FUNCTION SampledSpectrum sample(const BSDFQuery &query, const BSDFSample &sample, BSDFQueryResult* result) {
            if (!matches(query.dirTypeFilter)) {
                result->dirPDF = 0.0f;
                result->sampledType = DirectionType();
                return SampledSpectrum::Zero();
            }
            SampledSpectrum fs_sn = sampleInternal(query, sample.uComponent, sample.uDir, result);
            VLRAssert((result->dirPDF > 0 && fs_sn.allPositiveFinite()) || result->dirPDF == 0,
                      "Invalid BSDF value.\ndirV: (%g, %g, %g), sample: (%g, %g, %g), dirPDF: %g", 
                      query.dirLocal.x, query.dirLocal.y, query.dirLocal.z, sample.uComponent, sample.uDir[0], sample.uDir[1],
                      result->dirPDF);
            float snCorrection = std::fabs(result->dirLocal.z / dot(result->dirLocal, query.geometricNormalLocal));
            return fs_sn * snCorrection;
        }

        RT_FUNCTION SampledSpectrum evaluate(const BSDFQuery &query, const Vector3D &dirLocal) {
            SampledSpectrum fs_sn = evaluateInternal(query, dirLocal);
            float snCorrection = std::fabs(dirLocal.z / dot(dirLocal, query.geometricNormalLocal));
            return fs_sn * snCorrection;
        }

        RT_FUNCTION float evaluatePDF(const BSDFQuery &query, const Vector3D &dirLocal) {
            if (!matches(query.dirTypeFilter)) {
                return 0;
            }
            float ret = evaluatePDFInternal(query, dirLocal);
            return ret;
        }
    };



    class EDF {
#define VLR_MAX_NUM_EDF_PARAMETER_SLOTS (8)
        uint32_t data[VLR_MAX_NUM_EDF_PARAMETER_SLOTS];

        ProgSigEDFEvaluateEmittanceInternal progEvaluateEmittanceInternal;
        ProgSigEDFEvaluateInternal progEvaluateInternal;

        RT_FUNCTION SampledSpectrum evaluateEmittanceInternal() {
            return progEvaluateEmittanceInternal((const uint32_t*)this);
        }
        RT_FUNCTION SampledSpectrum evaluateInternal(const EDFQuery &query, const Vector3D &dirLocal) {
            return progEvaluateInternal((const uint32_t*)this, query, dirLocal);
        }

    public:
        RT_FUNCTION EDF(const SurfaceMaterialDescriptor &matDesc, const SurfacePoint &surfPt, const WavelengthSamples &wls) {
            ProgSigSetupEDF setupEDF = (ProgSigSetupEDF)matDesc.progSetupEDF;
            setupEDF(matDesc.data, surfPt, wls, (uint32_t*)this);

            const EDFProcedureSet procSet = pv_edfProcedureSetBuffer[matDesc.edfProcedureSetIndex];

            progEvaluateEmittanceInternal = (ProgSigEDFEvaluateEmittanceInternal)procSet.progEvaluateEmittanceInternal;
            progEvaluateInternal = (ProgSigEDFEvaluateInternal)procSet.progEvaluateInternal;
        }

        RT_FUNCTION SampledSpectrum evaluateEmittance() {
            SampledSpectrum Le0 = evaluateEmittanceInternal();
            return Le0;
        }

        RT_FUNCTION SampledSpectrum evaluate(const EDFQuery &query, const Vector3D &dirLocal) {
            SampledSpectrum Le1 = evaluateInternal(query, dirLocal);
            return Le1;
        }
    };



    struct Payload {
        struct {
            bool terminate : 1;
            bool maxLengthTerminate : 1;
        };
        KernelRNG rng;
        float initImportance;
        WavelengthSamples wls;
        SampledSpectrum alpha;
        SampledSpectrum contribution;
		//SampledSpectrum normal;
		//SampledSpectrum albedo;
        Point3D origin;
        Vector3D direction;
        float prevDirPDF;
        DirectionType prevSampledType;
    };

    struct ShadowPayload {
        KernelRNG rng;
        WavelengthSamples wls;
        float fractionalVisibility;
        SampledSpectrum shadow_color;
    };



    rtDeclareVariable(optix::uint2, sm_launchIndex, rtLaunchIndex, );
    rtDeclareVariable(Payload, sm_payload, rtPayload, );
    rtDeclareVariable(ShadowPayload, sm_shadowPayload, rtPayload, );

    typedef rtCallableProgramX<SampledSpectrum(const WavelengthSamples &, const LensPosSample &, LensPosQueryResult*)> ProgSigSampleLensPosition;
    typedef rtCallableProgramX<SampledSpectrum(const SurfacePoint &, const WavelengthSamples &, const IDFSample &, IDFQueryResult*)> ProgSigSampleIDF;

    typedef rtCallableProgramX<void(const HitPointParameter &, SurfacePoint*, float*)> ProgSigDecodeHitPoint;
    typedef rtCallableProgramX<float(const TexCoord2D &)> ProgSigFetchAlpha;
    typedef rtCallableProgramX<Normal3D(const TexCoord2D &)> ProgSigFetchNormal;

    // per GeometryInstance
    rtDeclareVariable(uint32_t, pv_geomInstIndex, , );
    rtDeclareVariable(ProgSigDecodeHitPoint, pv_progDecodeHitPoint, , );
    rtDeclareVariable(ShaderNodePlug, pv_nodeNormal, , );
    rtDeclareVariable(ShaderNodePlug, pv_nodeTangent, , );
    rtDeclareVariable(ShaderNodePlug, pv_nodeAlpha, , );
    rtDeclareVariable(uint32_t, pv_materialIndex, , );
    rtDeclareVariable(float, pv_importance, , );



    // Reference:
    // Chapter 6. A Fast and Robust Method for Avoiding Self-Intersection, Ray Tracing Gems, 2019
    RT_FUNCTION Point3D offsetRayOrigin(const Point3D &p, const Normal3D &geometricNormal) {
        constexpr float kOrigin = 1.0f / 32.0f;
        constexpr float kFloatScale = 1.0f / 65536.0f;
        constexpr float kIntScale = 256.0f;

        int32_t offsetInInt[] = {
            (int32_t)(kIntScale * geometricNormal.x),
            (int32_t)(kIntScale * geometricNormal.y),
            (int32_t)(kIntScale * geometricNormal.z)
        };

        // JP: 数学的な衝突点の座標と、実際の座標の誤差は原点からの距離に比例する。
        //     intとしてオフセットを加えることでスケール非依存に適切なオフセットを加えることができる。
        // EN: The error of the actual coorinates of the intersection point to the mathematical one is proportional to the distance to the origin.
        //     Applying the offset as int makes applying appropriate scale invariant amount of offset possible.
        Point3D newP1 = Point3D(__int_as_float(__float_as_int(p.x) + (p.x < 0 ? -1 : 1) * offsetInInt[0]),
                                __int_as_float(__float_as_int(p.y) + (p.y < 0 ? -1 : 1) * offsetInInt[1]),
                                __int_as_float(__float_as_int(p.z) + (p.z < 0 ? -1 : 1) * offsetInInt[2]));

        // JP: 原点に近い場所では、原点からの距離に依存せず一定の誤差が残るため別処理が必要。
        // EN: A constant amount of error remains near the origin independent of the distance to the origin so we need handle it separately.
        Point3D newP2 = p + kFloatScale * geometricNormal;

        return Point3D(std::fabs(p.x) < kOrigin ? newP2.x : newP1.x,
                       std::fabs(p.y) < kOrigin ? newP2.y : newP1.y,
                       std::fabs(p.z) < kOrigin ? newP2.z : newP1.z);
    }
	

    // JP: 変異された法線に従ってシェーディングフレームを変更する。
    // EN: perturb the shading frame according to the modified normal.
    RT_FUNCTION void applyBumpMapping(const Normal3D &modNormalInTF, SurfacePoint* surfPt) {
        if (modNormalInTF.x == 0.0f && modNormalInTF.y == 0.0f)
            return;

        // JP: 法線から回転軸と回転角(、Quaternion)を求めて対応する接平面ベクトルを求める。
        // EN: calculate a rotating axis and an angle (and quaternion) from the normal then calculate corresponding tangential vectors.
        float projLength = std::sqrt(modNormalInTF.x * modNormalInTF.x + modNormalInTF.y * modNormalInTF.y);
        float tiltAngle = std::atan(projLength / modNormalInTF.z);
        float qSin, qCos;
        VLR::sincos(tiltAngle / 2, &qSin, &qCos);
        float qX = (-modNormalInTF.y / projLength) * qSin;
        float qY = (modNormalInTF.x / projLength) * qSin;
        float qW = qCos;
        Vector3D modTangentInTF = Vector3D(1 - 2 * qY * qY, 2 * qX * qY, -2 * qY * qW);
        Vector3D modBitangentInTF = Vector3D(2 * qX * qY, 1 - 2 * qX * qX, 2 * qX * qW);

        Matrix3x3 matTFtoW = Matrix3x3(surfPt->shadingFrame.x, surfPt->shadingFrame.y, surfPt->shadingFrame.z);
        ReferenceFrame bumpShadingFrame(matTFtoW * modTangentInTF,
                                        matTFtoW * modBitangentInTF,
                                        matTFtoW * modNormalInTF);

        surfPt->shadingFrame = bumpShadingFrame;
    }



    // JP: 変異された接線に従ってシェーディングフレームを変更する。
    // EN: perturb the shading frame according to the modified tangent.
    RT_FUNCTION void modifyTangent(const Vector3D& modTangent, SurfacePoint* surfPt) {
        if (modTangent == surfPt->shadingFrame.x)
            return;

        float dotNT = dot(surfPt->shadingFrame.z, modTangent);
        Vector3D projModTangent = modTangent - dotNT * surfPt->shadingFrame.z;

        float lx = dot(surfPt->shadingFrame.x, projModTangent);
        float ly = dot(surfPt->shadingFrame.y, projModTangent);

        float tangentAngle = std::atan2(ly, lx);

        float s, c;
        VLR::sincos(tangentAngle, &s, &c);
        Vector3D modTangentInTF = Vector3D(c, s, 0);
        Vector3D modBitangentInTF = Vector3D(-s, c, 0);

        Matrix3x3 matTFtoW = Matrix3x3(surfPt->shadingFrame.x, surfPt->shadingFrame.y, surfPt->shadingFrame.z);
        ReferenceFrame newShadingFrame(normalize(matTFtoW * modTangentInTF),
                                       normalize(matTFtoW * modBitangentInTF),
                                       surfPt->shadingFrame.z);

        surfPt->shadingFrame = newShadingFrame;
    }



    RT_FUNCTION void calcSurfacePoint(SurfacePoint* surfPt, float* hypAreaPDF) {
        HitPointParameter hitPointParam = a_hitPointParam;
        pv_progDecodeHitPoint(hitPointParam, surfPt, hypAreaPDF);
        surfPt->geometryInstanceIndex = pv_geomInstIndex;

        Normal3D localNormal = calcNode(pv_nodeNormal, Normal3D(0.0f, 0.0f, 1.0f), *surfPt, sm_payload.wls);
        applyBumpMapping(localNormal, surfPt);

        Vector3D newTangent = calcNode(pv_nodeTangent, surfPt->shadingFrame.x, *surfPt, sm_payload.wls);
        modifyTangent(newTangent, surfPt);
    }


    // ----------------------------------------------------------------
    // Light

    RT_FUNCTION bool testVisibility(const SurfacePoint &shadingSurfacePoint, const SurfacePoint &lightSurfacePoint,
                                    Vector3D* shadowRayDir, float* squaredDistance, float* fractionalVisibility, SampledSpectrum *shadow_color) {
        VLRAssert(shadingSurfacePoint.atInfinity == false, "Shading point must be in finite region.");

        *shadowRayDir = lightSurfacePoint.calcDirectionFrom(shadingSurfacePoint.position, squaredDistance);

        const Normal3D &geomNormal = shadingSurfacePoint.geometricNormal;
        bool isFrontSide = dot(geomNormal, *shadowRayDir) > 0;
        Point3D shadingPoint = offsetRayOrigin(shadingSurfacePoint.position, isFrontSide ? geomNormal : -geomNormal);

        optix::Ray shadowRay = optix::make_Ray(asOptiXType(shadingPoint), asOptiXType(*shadowRayDir), RayType::Shadow, 0.0f, FLT_MAX);
        if (!lightSurfacePoint.atInfinity)
            shadowRay.tmax = std::sqrt(*squaredDistance) * 0.9999f;

        ShadowPayload shadowPayload;
        shadowPayload.rng = sm_payload.rng;
        shadowPayload.wls = sm_payload.wls;
        shadowPayload.fractionalVisibility = 1.0f;
        shadowPayload.shadow_color = SampledSpectrum::One();
        rtTrace(pv_topGroup, shadowRay, shadowPayload);

        *fractionalVisibility = shadowPayload.fractionalVisibility;	
        *shadow_color = shadowPayload.shadow_color;
		
		return *fractionalVisibility > 0;
    }

    RT_FUNCTION void selectSurfaceLight(float lightSample, SurfaceLight* light, float* lightProb, float* remapped) {
        float sumImps = pv_envLightDescriptor.importance + pv_lightImpDist.integral();
        float su = sumImps * lightSample;
        if (su < pv_envLightDescriptor.importance) {
            *light = SurfaceLight(pv_envLightDescriptor);
            *lightProb = pv_envLightDescriptor.importance / sumImps;
        }
        else {
            lightSample = (su - pv_envLightDescriptor.importance) / pv_lightImpDist.integral();
            uint32_t lightIdx = pv_lightImpDist.sample(lightSample, lightProb, remapped);
            *light = SurfaceLight(pv_geometryInstanceDescriptorBuffer[lightIdx]);
            *lightProb *= pv_lightImpDist.integral() / sumImps;
        }
    }

    RT_FUNCTION float getSumLightImportances() {
        return pv_envLightDescriptor.importance + pv_lightImpDist.integral();
    }

    RT_FUNCTION float evaluateEnvironmentAreaPDF(float phi, float theta) {
        VLRAssert(std::isfinite(phi) && std::isfinite(theta), "\"phi\", \"theta\": Not finite values %g, %g.", phi, theta);
        float uvPDF = pv_envLightDescriptor.body.asInfSphere.importanceMap.evaluatePDF(phi / (2 * M_PIf), theta / M_PIf);
        return uvPDF / (2 * M_PIf * M_PIf * std::sin(theta));
    }

    // END: Light
    // ----------------------------------------------------------------


    RT_FUNCTION float getAlpha() {
        HitPointParameter hitPointParam = a_hitPointParam;
        SurfacePoint surfPt;
        float hypAreaPDF;
        pv_progDecodeHitPoint(hitPointParam, &surfPt, &hypAreaPDF);

        return calcNode(pv_nodeAlpha, 1.0f, surfPt, sm_payload.wls);
    }
	   

    RT_PROGRAM void shadowAnyHitDefault() {
		sm_shadowPayload.shadow_color = SampledSpectrum::Zero();		
		sm_shadowPayload.fractionalVisibility = 0.0f;
		
        rtTerminateRay();
    }
    // Common Any Hit Program for All Primitive Types and Materials for non-shadow rays
    RT_PROGRAM void anyHitWithAlpha() {
        float alpha = getAlpha();
		alpha = 1.0f;
		
        // Stochastic Alpha Test
        if (sm_payload.rng.getFloat0cTo1o() >= alpha)
            rtIgnoreIntersection();
    }
	
    // Common Any Hit Program for All Primitive Types and Materials for shadow rays
    RT_PROGRAM void shadowAnyHitWithAlpha() {		
        float alpha = getAlpha();
        sm_shadowPayload.fractionalVisibility *= (1 - alpha);
		
        WavelengthSamples &wls = sm_shadowPayload.wls;
				
        SurfacePoint surfPt;
        float hypAreaPDF;
        calcSurfacePoint(&surfPt, &hypAreaPDF);

        const SurfaceMaterialDescriptor matDesc = pv_materialDescriptorBuffer[pv_materialIndex];
        BSDF bsdf(matDesc, surfPt, wls);

		const BSDFProcedureSet procSet = pv_bsdfProcedureSetBuffer[matDesc.bsdfProcedureSetIndex];
        auto progGetBaseColor = (ProgSigBSDFGetBaseColor)procSet.progGetBaseColor;
        SampledSpectrum fs = progGetBaseColor((const uint32_t *)&bsdf);
		//SampledSpectrum fs = SampledSpectrum(1.f, 1.f, 1.f);

		float nDi = fabs(dot(make_float3(surfPt.geometricNormal.x, surfPt.geometricNormal.y, surfPt.geometricNormal.z), sm_ray.direction));
		float3 attenuation = 1 - fresnel_schlick(nDi, 5, 1 - make_float3(fs.r, fs.g, fs.b), make_float3(1));
		sm_shadowPayload.shadow_color *= SampledSpectrum(attenuation.x, attenuation.y, attenuation.z);
		  
	//	sm_shadowPayload.shadow_color = SampledSpectrum(0,0,1);		

        if (sm_shadowPayload.fractionalVisibility <= 0.0f)
            rtTerminateRay();
        else
            rtIgnoreIntersection();
    }


}
