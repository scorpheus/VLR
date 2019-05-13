﻿#pragma once

#include "basic_types_internal.h"
#include "rgb_spectrum_types.h"
#include "spectrum_types.h"
#if defined(VLR_Host)
#include "../ext/include/half.hpp"
#endif

namespace VLR {
#if defined(VLR_Host)
    using half_float::half;
#else
    struct half {
        uint16_t raw;

        operator float() const {
            uint32_t bits = (uint32_t)(raw & 0x8000) << 16;
            uint32_t abs = raw & 0x7FFF;
            if (abs) {
                // JP: halfの指数部が   無限大 or NaN       を表す(11111)       場合: floatビット: (* 11100000 00000000000000000000000)
                //                    正規化数 or 非正規化数を表す(00000-11110) 場合:              (* 01110000 00000000000000000000000)
                bits |= 0x38000000 << (uint32_t)(abs >= 0x7C00);
                // JP: halfの指数部が非正規化数を表す(00000) 場合: 0x0001-0x03FF (* 00000 **********)
                //     正規化数になるまでhalfをビットシフト、floatの指数部を1ずつ減算。
                for (; abs < 0x400; abs <<= 1, bits -= 0x800000);
                // JP: halfの指数部が 無限大 or NaN を表す場合 0x7C00-0x7FFF (0       11111 **********): (0          00011111 **********0000000000000) を加算 => floatの指数ビットは0xFFになる。
                //                    正規化数      を表す場合 0x0400-0x7BFF (0 00001-11110 **********): (0 00000001-00011110 **********0000000000000) を加算 => floatの指数ビットは0x71-0x8Eになる。
                bits += (uint32_t)(abs) << 13;
            }
            return *(float*)&bits;
        }
    };
#endif

#if defined(VLR_USE_SPECTRAL_RENDERING)
    using WavelengthSamples = WavelengthSamplesTemplate<float, NumSpectralSamples>;
    using SampledSpectrum = SampledSpectrumTemplate<float, NumSpectralSamples>;
    using DiscretizedSpectrum = DiscretizedSpectrumTemplate<float, NumStrataForStorage>;
    using SpectrumStorage = SpectrumStorageTemplate<float, NumStrataForStorage>;
    using TripletSpectrum = UpsampledSpectrum;
#else
    using WavelengthSamples = RGBWavelengthSamplesTemplate<float>;
    using SampledSpectrum = RGBSpectrumTemplate<float>;
    using DiscretizedSpectrum = RGBSpectrumTemplate<float>;
    using SpectrumStorage = RGBStorageTemplate<float>;
    using TripletSpectrum = RGBSpectrum;
#endif

    using DiscretizedSpectrumAlwaysSpectral = DiscretizedSpectrumTemplate<float, NumStrataForStorage>;

#if defined(VLR_Device)
#   if SPECTRAL_UPSAMPLING_METHOD == MENG_SPECTRAL_UPSAMPLING
    rtDeclareVariable(int32_t, UpsampledSpectrum_spectrum_grid, , );
    rtDeclareVariable(int32_t, UpsampledSpectrum_spectrum_data_points, , );
#   elif SPECTRAL_UPSAMPLING_METHOD == JAKOB_SPECTRAL_UPSAMPLING
    rtDeclareVariable(int32_t, UpsampledSpectrum_maxBrightnesses, , );
    rtDeclareVariable(int32_t, UpsampledSpectrum_coefficients_sRGB_D65, , );
    rtDeclareVariable(int32_t, UpsampledSpectrum_coefficients_sRGB_E, , );
#   endif

    rtDeclareVariable(DiscretizedSpectrumAlwaysSpectral::CMF, DiscretizedSpectrum_xbar, , );
    rtDeclareVariable(DiscretizedSpectrumAlwaysSpectral::CMF, DiscretizedSpectrum_ybar, , );
    rtDeclareVariable(DiscretizedSpectrumAlwaysSpectral::CMF, DiscretizedSpectrum_zbar, , );
    rtDeclareVariable(float, DiscretizedSpectrum_integralCMF, , );
#endif

    RT_FUNCTION HOST_INLINE TripletSpectrum createTripletSpectrum(SpectrumType spectrumType, ColorSpace colorSpace, float e0, float e1, float e2) {
#if defined(VLR_USE_SPECTRAL_RENDERING)
        return UpsampledSpectrum(spectrumType, colorSpace, e0, e1, e2);
#else
        float XYZ[3];

        switch (colorSpace) {
        case ColorSpace::Rec709_D65_sRGBGamma: {
            e0 = sRGB_degamma(e0);
            e1 = sRGB_degamma(e1);
            e2 = sRGB_degamma(e2);
            // pass to Rec709 (D65)
        }
        case ColorSpace::Rec709_D65: {
            float RGB[3] = { e0, e1, e2 };
            switch (spectrumType) {
            case SpectrumType::Reflectance:
            case SpectrumType::IndexOfRefraction:
            case SpectrumType::NA:
                transformTristimulus(mat_Rec709_E_to_XYZ, RGB, XYZ);
                break;
            case SpectrumType::LightSource:
                transformTristimulus(mat_Rec709_D65_to_XYZ, RGB, XYZ);
                break;
            default:
                VLRAssert_ShouldNotBeCalled();
                break;
            }
            break;
        }
        case ColorSpace::XYZ: {
            XYZ[0] = e0;
            XYZ[1] = e1;
            XYZ[2] = e2;
            break;
        }
        case ColorSpace::xyY: {
            VLRAssert(e0 >= 0.0f && e1 >= 0.0f && e0 <= 1.0f && e1 <= 1.0f && e2 >= 0.0f,
                      "xy should be in [0, 1], Y should not be negative.");
            if (e1 == 0) {
                XYZ[0] = XYZ[1] = XYZ[2] = 0;
                break;
            }
            float z = 1 - (e0 + e1);
            float b = e2 / e1;
            XYZ[0] = e0 * b;
            XYZ[1] = e2;
            XYZ[2] = z * b;
            break;
        }
        default:
            VLRAssert_NotImplemented();
            break;
        }

        float RGB[3];
        transformToRenderingRGB(spectrumType, XYZ, RGB);
        return RGBSpectrum(RGB[0], RGB[1], RGB[2]);
#endif
    }



    enum class DataFormat {
        RGB8x3 = 0,
        RGB_8x4,
        RGBA8x4,
        RGBA16Fx4,
        RGBA32Fx4,
        RG32Fx2,
        Gray32F,
        Gray8,
        GrayA8x2,
        BC1,
        BC2,
        BC3,
        BC4,
        BC4_Signed,
        BC5,
        BC5_Signed,
        BC6H,
        BC6H_Signed,
        BC7,
        // ---- Internal Formats ----
        uvsA8x4,
        uvsA16Fx4,
        NumFormats
    };



    enum class ShaderNodeSocketType {
        float1 = 0,
        float2,
        float3,
        float4,
        Point3D,
        Vector3D,
        Normal3D,
        Spectrum,
        Alpha,
        TextureCoordinates,
        NumTypes
    };



    namespace Shared {
        template <typename RealType>
        class DiscreteDistribution1DTemplate {
            rtBufferId<RealType, 1> m_PMF;
            rtBufferId<RealType, 1> m_CDF;
            RealType m_integral;
            uint32_t m_numValues;

        public:
            DiscreteDistribution1DTemplate(const rtBufferId<RealType, 1> &PMF, const rtBufferId<RealType, 1> &CDF, RealType integral, uint32_t numValues) : 
            m_PMF(PMF), m_CDF(CDF), m_integral(integral), m_numValues(numValues) {
            }

            RT_FUNCTION DiscreteDistribution1DTemplate() {}
            RT_FUNCTION ~DiscreteDistribution1DTemplate() {}

            RT_FUNCTION uint32_t sample(RealType u, RealType* prob) const {
                VLRAssert(u >= 0 && u < 1, "\"u\": %g must be in range [0, 1).", u);
                int idx = m_numValues;
                for (int d = prevPowerOf2(m_numValues); d > 0; d >>= 1) {
                    int newIdx = idx - d;
                    if (newIdx > 0 && m_CDF[newIdx] > u)
                        idx = newIdx;
                }
                --idx;
                VLRAssert(idx >= 0 && idx < m_numValues, "Invalid Index!: %d", idx);
                *prob = m_PMF[idx];
                return idx;
            }
            RT_FUNCTION uint32_t sample(RealType u, RealType* prob, RealType* remapped) const {
                VLRAssert(u >= 0 && u < 1, "\"u\": %g must be in range [0, 1).", u);
                int idx = m_numValues;
                for (int d = prevPowerOf2(m_numValues); d > 0; d >>= 1) {
                    int newIdx = idx - d;
                    if (newIdx > 0 && m_CDF[newIdx] > u)
                        idx = newIdx;
                }
                --idx;
                VLRAssert(idx >= 0 && idx < m_numValues, "Invalid Index!: %d", idx);
                *prob = m_PMF[idx];
                *remapped = (u - m_CDF[idx]) / (m_CDF[idx + 1] - m_CDF[idx]);
                return idx;
            }
            RT_FUNCTION RealType evaluatePMF(uint32_t idx) const {
                VLRAssert(idx >= 0 && idx < m_numValues, "\"idx\" is out of range [0, %u)", m_numValues);
                return m_PMF[idx];
            }

            RT_FUNCTION RealType integral() const { return m_integral; }
            RT_FUNCTION uint32_t numValues() const { return m_numValues; }
        };

        using DiscreteDistribution1D = DiscreteDistribution1DTemplate<float>;



        template <typename RealType>
        class RegularConstantContinuousDistribution1DTemplate {
            rtBufferId<RealType, 1> m_PDF;
            rtBufferId<RealType, 1> m_CDF;
            RealType m_integral;
            uint32_t m_numValues;

        public:
            RegularConstantContinuousDistribution1DTemplate(const rtBufferId<RealType, 1> &PDF, const rtBufferId<RealType, 1> &CDF, RealType integral, uint32_t numValues) :
                m_PDF(PDF), m_CDF(CDF), m_integral(integral), m_numValues(numValues) {
            }

            RT_FUNCTION RegularConstantContinuousDistribution1DTemplate() {}
            RT_FUNCTION ~RegularConstantContinuousDistribution1DTemplate() {}

            RT_FUNCTION RealType sample(RealType u, RealType* probDensity) const {
                VLRAssert(u < 1, "\"u\": %g must be in range [0, 1).", u);
                int idx = m_numValues;
                for (int d = prevPowerOf2(m_numValues); d > 0; d >>= 1) {
                    int newIdx = idx - d;
                    if (newIdx > 0 && m_CDF[newIdx] > u)
                        idx = newIdx;
                }
                --idx;
                VLRAssert(idx >= 0 && idx < m_numValues, "Invalid Index!: %d", idx);
                *probDensity = m_PDF[idx];
                RealType t = (u - m_CDF[idx]) / (m_CDF[idx + 1] - m_CDF[idx]);
                return (idx + t) / m_numValues;
            }
            RT_FUNCTION RealType evaluatePDF(RealType smp) const {
                VLRAssert(smp >= 0 && smp < 1.0, "\"smp\": %g is out of range [0, 1).", smp);
                int32_t idx = std::min<int32_t>(m_numValues - 1, smp * m_numValues);
                return m_PDF[idx];
            }
            RT_FUNCTION RealType integral() const { return m_integral; }

            RT_FUNCTION uint32_t numValues() const { return m_numValues; }
        };

        using RegularConstantContinuousDistribution1D = RegularConstantContinuousDistribution1DTemplate<float>;



        template <typename RealType>
        class RegularConstantContinuousDistribution2DTemplate {
            rtBufferId<RegularConstantContinuousDistribution1DTemplate<RealType>, 1> m_1DDists;
            RegularConstantContinuousDistribution1DTemplate<RealType> m_top1DDist;

        public:
            RegularConstantContinuousDistribution2DTemplate(const rtBufferId<RegularConstantContinuousDistribution1DTemplate<RealType>, 1> &_1DDists, 
                                                            const RegularConstantContinuousDistribution1DTemplate<RealType> &top1DDist) :
                m_1DDists(_1DDists), m_top1DDist(top1DDist) {
            }

            RT_FUNCTION RegularConstantContinuousDistribution2DTemplate() {}
            RT_FUNCTION ~RegularConstantContinuousDistribution2DTemplate() {}

            RT_FUNCTION void sample(RealType u0, RealType u1, RealType* d0, RealType* d1, RealType* probDensity) const {
                RealType topPDF;
                *d1 = m_top1DDist.sample(u1, &topPDF);
                uint32_t idx1D = std::min(uint32_t(m_top1DDist.numValues() * *d1), m_top1DDist.numValues() - 1);
                *d0 = m_1DDists[idx1D].sample(u0, probDensity);
                *probDensity *= topPDF;
            }
            RT_FUNCTION RealType evaluatePDF(RealType d0, RealType d1) const {
                uint32_t idx1D = std::min(uint32_t(m_top1DDist.numValues() * d1), m_top1DDist.numValues() - 1);
                return m_top1DDist.evaluatePDF(d1) * m_1DDists[idx1D].evaluatePDF(d0);
            }
        };

        using RegularConstantContinuousDistribution2D = RegularConstantContinuousDistribution2DTemplate<float>;



        class StaticTransform {
            Matrix4x4 m_matrix;
            Matrix4x4 m_invMatrix;

        public:
            RT_FUNCTION StaticTransform(const Matrix4x4 &m = Matrix4x4::Identity()) : m_matrix(m), m_invMatrix(invert(m)) {}

            RT_FUNCTION Vector3D operator*(const Vector3D &v) const { return m_matrix * v; }
            RT_FUNCTION Vector4D operator*(const Vector4D &v) const { return m_matrix * v; }
            RT_FUNCTION Point3D operator*(const Point3D &p) const { return m_matrix * p; }
            RT_FUNCTION Normal3D operator*(const Normal3D &n) const {
                // The length of the normal is changed if the transform has scaling, so it requires normalization.
                return Normal3D(m_invMatrix.m00 * n.x + m_invMatrix.m10 * n.y + m_invMatrix.m20 * n.z,
                                m_invMatrix.m01 * n.x + m_invMatrix.m11 * n.y + m_invMatrix.m21 * n.z,
                                m_invMatrix.m02 * n.x + m_invMatrix.m12 * n.y + m_invMatrix.m22 * n.z);
            }

            RT_FUNCTION StaticTransform operator*(const Matrix4x4 &m) const { return StaticTransform(m_matrix * m); }
            RT_FUNCTION StaticTransform operator*(const StaticTransform &t) const { return StaticTransform(m_matrix * t.m_matrix); }
            RT_FUNCTION bool operator==(const StaticTransform &t) const { return m_matrix == t.m_matrix; }
            RT_FUNCTION bool operator!=(const StaticTransform &t) const { return m_matrix != t.m_matrix; }
        };



        struct NodeProcedureSet {
            int32_t progs[nextPowerOf2((uint32_t)ShaderNodeSocketType::NumTypes)];
        };



        union ShaderNodeSocket {
            struct {
                unsigned int nodeType : 8;
                unsigned int socketType : 4;
                unsigned int nodeDescIndex : 18;
                unsigned int option : 2;
            };
            uint32_t asUInt;

            RT_FUNCTION ShaderNodeSocket() {}
            explicit constexpr ShaderNodeSocket(uint32_t ui) : asUInt(ui) {}
            RT_FUNCTION bool isValid() const { return asUInt != 0xFFFFFFFF; }

            static constexpr ShaderNodeSocket Invalid() { return ShaderNodeSocket(0xFFFFFFFF); }
        };
        static_assert(sizeof(ShaderNodeSocket) == 4, "Unexpected Size");

        struct SmallNodeDescriptor {
#define VLR_MAX_NUM_SMALL_NODE_DESCRIPTOR_SLOTS (4)
            uint32_t data[VLR_MAX_NUM_SMALL_NODE_DESCRIPTOR_SLOTS];

            template <typename T>
            RT_FUNCTION T* getData() const {
                VLRAssert(sizeof(T) <= sizeof(data), "Too big node data.");
                return (T*)data;
            }
        };

        struct MediumNodeDescriptor {
#define VLR_MAX_NUM_MEDIUM_NODE_DESCRIPTOR_SLOTS (16)
            uint32_t data[VLR_MAX_NUM_MEDIUM_NODE_DESCRIPTOR_SLOTS];

            template <typename T>
            RT_FUNCTION T* getData() const {
                VLRAssert(sizeof(T) <= sizeof(data), "Too big node data.");
                return (T*)data;
            }
        };

        struct LargeNodeDescriptor {
#   define VLR_MAX_NUM_LARGE_NODE_DESCRIPTOR_SLOTS (64)
            uint32_t data[VLR_MAX_NUM_LARGE_NODE_DESCRIPTOR_SLOTS];

            template <typename T>
            RT_FUNCTION T* getData() const {
                VLRAssert(sizeof(T) <= sizeof(data), "Too big node data.");
                return (T*)data;
            }
        };



        // ----------------------------------------------------------------
        // JP: シェーダーノードソケット間の暗黙的な型キャストを定義する。
        // EN: Define implicit type casting between shader node sockets.
        
        template <typename Type>
        struct NodeTypeInfo {
            template <typename SrcType>
            RT_FUNCTION static constexpr bool ConversionIsDefinedFor() {
                return false;
            }
            RT_FUNCTION static constexpr bool ConversionIsDefinedFor(ShaderNodeSocketType socketType);
            template <typename SrcType>
            RT_FUNCTION static Type convertFrom(const SrcType &) {
                return Type();
            }
        };

#define VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(DstType, SrcType) \
    template <> template <> HOST_INLINE constexpr bool NodeTypeInfo<DstType>::ConversionIsDefinedFor<SrcType>() { return true; } \
    template <> template <> HOST_INLINE DstType NodeTypeInfo<DstType>::convertFrom<SrcType>(const SrcType &srcValue)

        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(float, float) { return srcValue; }

        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(optix::float2, optix::float2) { return srcValue; }
        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(optix::float2, float) { return optix::make_float2(srcValue); }

        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(optix::float3, optix::float3) { return srcValue; }
        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(optix::float3, float) { return optix::make_float3(srcValue); }
        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(optix::float3, Point3D) { return asOptiXType(srcValue); }
        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(optix::float3, Vector3D) { return asOptiXType(srcValue); }
        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(optix::float3, Normal3D) { return asOptiXType(srcValue); }

        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(optix::float4, optix::float4) { return srcValue; }
        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(optix::float4, float) { return optix::make_float4(srcValue); }

        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(Point3D, Point3D) { return srcValue; }
        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(Point3D, optix::float3) { return asPoint3D(srcValue); }
        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(Point3D, Vector3D) { return Point3D(srcValue.x, srcValue.y, srcValue.z); }
        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(Point3D, Normal3D) { return Point3D(srcValue.x, srcValue.y, srcValue.z); }

        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(Vector3D, Vector3D) { return srcValue; }
        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(Vector3D, optix::float3) { return asVector3D(srcValue); }
        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(Vector3D, Point3D) { return Vector3D(srcValue.x, srcValue.y, srcValue.z); }
        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(Vector3D, Normal3D) { return Vector3D(srcValue.x, srcValue.y, srcValue.z); }

        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(Normal3D, Normal3D) { return srcValue; }
        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(Normal3D, optix::float3) { return asNormal3D(srcValue).normalize(); }
        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(Normal3D, Point3D) { return Normal3D(srcValue.x, srcValue.y, srcValue.z).normalize(); }
        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(Normal3D, Vector3D) { return Normal3D(srcValue.x, srcValue.y, srcValue.z).normalize(); }

        VLR_NODE_TYPE_INFO_DEFINE_CONVERSION(SampledSpectrum, SampledSpectrum) { return srcValue; }

#undef VLR_NODE_TYPE_INFO_DEFINE_CONVERSION

        template <typename Type>
        RT_FUNCTION constexpr bool NodeTypeInfo<Type>::ConversionIsDefinedFor(ShaderNodeSocketType socketType) {
            switch (socketType) {
            case ShaderNodeSocketType::float1:
                return ConversionIsDefinedFor<float>();
            case ShaderNodeSocketType::float2:
                return ConversionIsDefinedFor<optix::float2>();
            case ShaderNodeSocketType::float3:
                return ConversionIsDefinedFor<optix::float3>();
            case ShaderNodeSocketType::float4:
                return ConversionIsDefinedFor<optix::float4>();
            case ShaderNodeSocketType::Point3D:
                return ConversionIsDefinedFor<Point3D>();
            case ShaderNodeSocketType::Vector3D:
                return ConversionIsDefinedFor<Vector3D>();
            case ShaderNodeSocketType::Normal3D:
                return ConversionIsDefinedFor<Normal3D>();
            case ShaderNodeSocketType::Spectrum:
                return ConversionIsDefinedFor<SampledSpectrum>();
            case ShaderNodeSocketType::Alpha:
                return ConversionIsDefinedFor<float>();
            case ShaderNodeSocketType::TextureCoordinates:
                return ConversionIsDefinedFor<Point3D>();
            default:
                VLRAssert_ShouldNotBeCalled();
                break;
            }
            return false;
        }

        // END: Define implicit type casting between shader node sockets.
        // ----------------------------------------------------------------



        struct BSDFProcedureSet {
            int32_t progGetBaseColor;
            int32_t progMatches;
            int32_t progSampleInternal;
            int32_t progEvaluateInternal;
            int32_t progEvaluatePDFInternal;
            int32_t progWeightInternal;
        };

        struct EDFProcedureSet {
            int32_t progEvaluateEmittanceInternal;
            int32_t progEvaluateInternal;
        };



        struct SurfaceMaterialDescriptor {
            int32_t progSetupBSDF;
            uint32_t bsdfProcedureSetIndex;
            int32_t progSetupEDF;
            uint32_t edfProcedureSetIndex;
#define VLR_MAX_NUM_MATERIAL_DESCRIPTOR_SLOTS (28)
            uint32_t data[VLR_MAX_NUM_MATERIAL_DESCRIPTOR_SLOTS];

            template <typename T>
            T* getData() const {
                static_assert(sizeof(T) <= sizeof(data), "Too big node data.");
                return (T*)data;
            }
        };



        struct Triangle {
            uint32_t index0, index1, index2;
        };

        struct SurfaceLightDescriptor {
            union Body {
                struct {
                    rtBufferId<Vertex> vertexBuffer;
                    rtBufferId<Triangle> triangleBuffer;
                    uint32_t materialIndex;
                    DiscreteDistribution1D primDistribution;
                    StaticTransform transform;
                } asMeshLight;
                struct {
                    uint32_t materialIndex;
                    float rotationPhi;
                    RegularConstantContinuousDistribution2D importanceMap;
                } asEnvironmentLight;

                RT_FUNCTION Body() {}
                RT_FUNCTION ~Body() {}
            } body;
            float importance;
            int32_t sampleFunc;
        };



        struct PerspectiveCamera {
            Point3D position;
            Quaternion orientation;

            float sensitivity;
            float aspect;
            float fovY;
            float lensRadius;
            float imgPlaneDistance;
            float objPlaneDistance;

            float opWidth;
            float opHeight;
            float imgPlaneArea;

            RT_FUNCTION PerspectiveCamera() {}

            void setImagePlaneArea() {
                opHeight = 2.0f * objPlaneDistance * std::tan(fovY * 0.5f);
                opWidth = opHeight * aspect;
                imgPlaneDistance = 1.0f;
                imgPlaneArea = 1;// opWidth * opHeight * std::pow(imgPlaneDistance / objPlaneDistance, 2);
            }
        };



        struct EquirectangularCamera {
            Point3D position;
            Quaternion orientation;

            float sensitivity;

            float phiAngle;
            float thetaAngle;

            RT_FUNCTION EquirectangularCamera() {}
        };



        struct RayType {
            enum Value {
                Primary = 0,
                Scattered,
                Shadow,
                DebugPrimary,
                NumTypes
            } value;

            RT_FUNCTION constexpr RayType(Value v = Primary) : value(v) {}
        };

        struct DebugRenderingAttribute {
            enum Value {
                BaseColor = 0,
                GeometricNormal,
                ShadingTangent,
                ShadingBitangent,
                ShadingNormal,
                TC0Direction,
                TextureCoordinates,
                GeometricVsShadingNormal,
                ShadingFrameLengths,
                ShadingFrameOrthogonality,
            } value;

            RT_FUNCTION constexpr DebugRenderingAttribute(Value v = BaseColor) : value(v) {}

            RT_FUNCTION operator int32_t() const {
                return value;
            }
        };



        struct TangentType {
            enum Value {
                TC0Direction = 0,
                RadialX,
                RadialY,
                RadialZ
            } value;

            RT_FUNCTION constexpr TangentType(Value v = TC0Direction) : value(v) {}
            RT_FUNCTION bool operator==(const TangentType &r) const {
                return value == r.value;
            }
        };



        // ----------------------------------------------------------------
        // Shader Nodes

        struct GeometryShaderNode {

        };

        struct FloatShaderNode {
            ShaderNodeSocket node0;
            float imm0;
        };

        struct Float2ShaderNode {
            ShaderNodeSocket node0;
            ShaderNodeSocket node1;
            float imm0;
            float imm1;
        };

        struct Float3ShaderNode {
            ShaderNodeSocket node0;
            ShaderNodeSocket node1;
            ShaderNodeSocket node2;
            float imm0;
            float imm1;
            float imm2;
        };

        struct Float4ShaderNode {
            ShaderNodeSocket node0;
            ShaderNodeSocket node1;
            ShaderNodeSocket node2;
            ShaderNodeSocket node3;
            float imm0;
            float imm1;
            float imm2;
            float imm3;
        };

        struct ScaleAndOffsetFloatShaderNode {
            ShaderNodeSocket nodeValue;
            ShaderNodeSocket nodeScale;
            ShaderNodeSocket nodeOffset;
            float immScale;
            float immOffset;
        };

#if defined(VLR_USE_SPECTRAL_RENDERING)
        struct TripletSpectrumShaderNode {
            UpsampledSpectrum value;
        };

        struct RegularSampledSpectrumShaderNode {
            float minLambda;
            float maxLambda;
            float values[VLR_MAX_NUM_LARGE_NODE_DESCRIPTOR_SLOTS - 3];
            uint32_t numSamples;
        };

        struct IrregularSampledSpectrumShaderNode {
            float lambdas[(VLR_MAX_NUM_LARGE_NODE_DESCRIPTOR_SLOTS - 1) / 2];
            float values[(VLR_MAX_NUM_LARGE_NODE_DESCRIPTOR_SLOTS - 1) / 2];
            uint32_t numSamples;
        };
#else
        struct TripletSpectrumShaderNode {
            RGBSpectrum value;
        };

        struct RegularSampledSpectrumShaderNode {
            RGBSpectrum value;
        };

        struct IrregularSampledSpectrumShaderNode {
            RGBSpectrum value;
        };
#endif

        struct Float3ToSpectrumShaderNode {
            ShaderNodeSocket nodeFloat3;
            float immFloat3[3];
            SpectrumType spectrumType;
            ColorSpace colorSpace;
        };

        struct ScaleAndOffsetUVTextureMap2DShaderNode {
            float offset[2];
            float scale[2];
        };

        struct Image2DTextureShaderNode {
            int32_t textureID;
            struct {
                unsigned int dataFormat : 5;
                unsigned int spectrumType : 3;
                unsigned int colorSpace : 3;
            };
            ShaderNodeSocket nodeTexCoord;

            RT_FUNCTION DataFormat getDataFormat() const { return DataFormat(dataFormat); }
            RT_FUNCTION SpectrumType getSpectrumType() const { return SpectrumType(spectrumType); }
            RT_FUNCTION ColorSpace getColorSpace() const { return ColorSpace(colorSpace); }
        };
        static_assert(sizeof(Image2DTextureShaderNode) == 12, "Unexpected sizeof(Image2DTextureShaderNode).");

        struct EnvironmentTextureShaderNode {
            int32_t textureID;
            struct {
                unsigned int dataFormat : 5;
                unsigned int colorSpace : 3;
            };
            ShaderNodeSocket nodeTexCoord;

            RT_FUNCTION DataFormat getDataFormat() const { return DataFormat(dataFormat); }
            RT_FUNCTION ColorSpace getColorSpace() const { return ColorSpace(colorSpace); }
        };

        // END: Shader Nodes
        // ----------------------------------------------------------------



        // ----------------------------------------------------------------
        // Surface Materials

        struct MatteSurfaceMaterial {
            ShaderNodeSocket nodeAlbedo;
            TripletSpectrum immAlbedo;
        };

        struct SpecularReflectionSurfaceMaterial {
            ShaderNodeSocket nodeCoeffR;
            ShaderNodeSocket nodeEta;
            ShaderNodeSocket node_k;
            TripletSpectrum immCoeffR;
            TripletSpectrum immEta;
            TripletSpectrum imm_k;
        };

        struct SpecularScatteringSurfaceMaterial {
            ShaderNodeSocket nodeCoeff;
            ShaderNodeSocket nodeEtaExt;
            ShaderNodeSocket nodeEtaInt;
            TripletSpectrum immCoeff;
            TripletSpectrum immEtaExt;
            TripletSpectrum immEtaInt;
        };

        struct MicrofacetReflectionSurfaceMaterial {
            ShaderNodeSocket nodeEta;
            ShaderNodeSocket node_k;
            ShaderNodeSocket nodeRoughnessAnisotropyRotation;
            TripletSpectrum immEta;
            TripletSpectrum imm_k;
            float immRoughness;
            float immAnisotropy;
            float immRotation;
        };

        struct MicrofacetScatteringSurfaceMaterial {
            ShaderNodeSocket nodeCoeff;
            ShaderNodeSocket nodeEtaExt;
            ShaderNodeSocket nodeEtaInt;
            ShaderNodeSocket nodeRoughnessAnisotropyRotation;
            TripletSpectrum immCoeff;
            TripletSpectrum immEtaExt;
            TripletSpectrum immEtaInt;
            float immRoughness;
            float immAnisotropy;
            float immRotation;
        };

        struct LambertianScatteringSurfaceMaterial {
            ShaderNodeSocket nodeCoeff;
            ShaderNodeSocket nodeF0;
            TripletSpectrum immCoeff;
            float immF0;
        };

        struct UE4SurfaceMaterial {
            ShaderNodeSocket nodeBaseColor;
            ShaderNodeSocket nodeOcclusionRoughnessMetallic;
            TripletSpectrum immBaseColor;
            float immOcclusion;
            float immRoughness;
            float immMetallic;
        };

        struct OldStyleSurfaceMaterial {
            ShaderNodeSocket nodeDiffuseColor;
            ShaderNodeSocket nodeSpecularColor;
            ShaderNodeSocket nodeGlossiness;
            TripletSpectrum immDiffuseColor;
            TripletSpectrum immSpecularColor;
            float immGlossiness;
        };

        struct DiffuseEmitterSurfaceMaterial {
            ShaderNodeSocket nodeEmittance;
            TripletSpectrum immEmittance;
            float immScale;
        };

        struct MultiSurfaceMaterial {
            uint32_t subMatIndices[4];
            uint32_t numSubMaterials;
        };

        struct EnvironmentEmitterSurfaceMaterial {
            ShaderNodeSocket nodeEmittance;
            TripletSpectrum immEmittance;
            float immScale;
        };

        // END: Surface Materials
        // ----------------------------------------------------------------
    }
}

#if defined(VLR_Device)
#include "spectrum_types.cpp"
#endif
