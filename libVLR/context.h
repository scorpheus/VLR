﻿#pragma once

#include <public_types.h>
#include "shared/shared.h"

#include "slot_finder.h"

#if defined(_DEBUG)
#   define VLR_PTX_DIR "resources/ptxes/Debug/"
#else
#   define VLR_PTX_DIR "resources/ptxes/Release/"
#endif

namespace VLR {
    std::string readTxtFile(const filesystem::path& filepath);



    enum class TextureFilter {
        Nearest = 0,
        Linear,
        None
    };

    enum class TextureWrapMode {
        Repeat = 0,
        ClampToEdge,
        Mirror,
        ClampToBorder,
    };



    class Scene;
    class Camera;

    template <typename InternalType>
    struct SlotBuffer {
        uint32_t maxNumElements;
        optix::Buffer optixBuffer;
        SlotFinder slotFinder;

        void initialize(optix::Context &context, uint32_t _maxNumElements, const char* varName) {
            maxNumElements = _maxNumElements;
            optixBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, maxNumElements);
            optixBuffer->setElementSize(sizeof(InternalType));
            slotFinder.initialize(maxNumElements);
            if (varName)
                context[varName]->set(optixBuffer);
        }
        void finalize() {
            slotFinder.finalize();
            optixBuffer->destroy();
        }

        uint32_t allocate() {
            uint32_t index = slotFinder.getFirstAvailableSlot();
            slotFinder.setInUse(index);
            return index;
        }

        void release(uint32_t index) {
            VLRAssert(slotFinder.getUsage(index), "Invalid index.");
            slotFinder.setNotInUse(index);
        }

        void get(uint32_t index, InternalType* value) {
            VLRAssert(slotFinder.getUsage(index), "Invalid index.");
            auto values = (InternalType*)optixBuffer->map(0, RT_BUFFER_MAP_READ);
            *value = values[index];
            optixBuffer->unmap();
        }

        void update(uint32_t index, const InternalType &value) {
            VLRAssert(slotFinder.getUsage(index), "Invalid index.");
            auto values = (InternalType*)optixBuffer->map(0, RT_BUFFER_MAP_WRITE);
            values[index] = value;
            optixBuffer->unmap();
        }
    };



    class Context {
        static uint32_t NextID;
        static uint32_t getInstanceID() {
            return NextID++;
        }

        uint32_t m_ID;
        optix::Context m_optixContext;
        bool m_RTXEnabled;
        int32_t* m_devices;
        uint32_t m_numDevices;

        optix::Program m_optixProgramShadowAnyHitDefault; // ---- Any Hit Program
        optix::Program m_optixProgramAnyHitWithAlpha; // -------- Any Hit Program
        optix::Program m_optixProgramShadowAnyHitWithAlpha; // -- Any Hit Program
        optix::Program m_optixProgramPathTracingIteration; // --- Closest Hit Program

        optix::Program m_optixProgramPathTracing; // ------------ Ray Generation Program
        optix::Program m_optixProgramPathTracingMiss; // -------- Miss Program
        optix::Program m_optixProgramException; // -------------- Exception Program

        optix::Program m_optixProgramDebugRenderingClosestHit;
        optix::Program m_optixProgramDebugRenderingAnyHitWithAlpha;
        optix::Program m_optixProgramDebugRenderingMiss;
        optix::Program m_optixProgramDebugRenderingRayGeneration;
        optix::Program m_optixProgramDebugRenderingException;

        optix::Program m_optixProgramConvertToRGB; // ----------- Ray Generation Program (TODO: port to pure CUDA code)

#if SPECTRAL_UPSAMPLING_METHOD == MENG_SPECTRAL_UPSAMPLING
        optix::Buffer m_optixBufferUpsampledSpectrum_spectrum_grid;
        optix::Buffer m_optixBufferUpsampledSpectrum_spectrum_data_points;
#elif SPECTRAL_UPSAMPLING_METHOD == JAKOB_SPECTRAL_UPSAMPLING
        optix::Buffer m_optixBufferUpsampledSpectrum_maxBrightnesses;
        optix::Buffer m_optixBufferUpsampledSpectrum_coefficients_sRGB_D65;
        optix::Buffer m_optixBufferUpsampledSpectrum_coefficients_sRGB_E;
#endif

        optix::Material m_optixMaterialDefault;
        optix::Material m_optixMaterialWithAlpha;

        SlotBuffer<Shared::NodeProcedureSet> m_nodeProcedureBuffer;

        SlotBuffer<Shared::SmallNodeDescriptor> m_smallNodeDescriptorBuffer;
        SlotBuffer<Shared::MediumNodeDescriptor> m_mediumNodeDescriptorBuffer;
        SlotBuffer<Shared::LargeNodeDescriptor> m_largeNodeDescriptorBuffer;

        SlotBuffer<Shared::BSDFProcedureSet> m_BSDFProcedureBuffer;
        SlotBuffer<Shared::EDFProcedureSet> m_EDFProcedureBuffer;

        optix::Program m_optixCallableProgramNullBSDF_setupBSDF;
        optix::Program m_optixCallableProgramNullBSDF_getBaseColor;
        optix::Program m_optixCallableProgramNullBSDF_matches;
        optix::Program m_optixCallableProgramNullBSDF_sampleInternal;
        optix::Program m_optixCallableProgramNullBSDF_evaluateInternal;
        optix::Program m_optixCallableProgramNullBSDF_evaluatePDFInternal;
        optix::Program m_optixCallableProgramNullBSDF_weightInternal;
        uint32_t m_nullBSDFProcedureSetIndex;

        optix::Program m_optixCallableProgramNullEDF_setupEDF;
        optix::Program m_optixCallableProgramNullEDF_evaluateEmittanceInternal;
        optix::Program m_optixCallableProgramNullEDF_evaluateInternal;
        uint32_t m_nullEDFProcedureSetIndex;

        SlotBuffer<Shared::SurfaceMaterialDescriptor> m_surfaceMaterialDescriptorBuffer;

		optix::PostprocessingStage m_denoiserStage;
		optix::Buffer trainingDataBuffer;
				
		optix::CommandList commandListWithDenoiser, commandListNoDenoiser;

        optix::Buffer m_rawOutputBuffer;
        optix::Buffer m_outputBuffer;
		optix::Buffer m_outputBufferDenoise;
		optix::Buffer m_rngBuffer;
		optix::Buffer m_outputNormalBuffer;
		optix::Buffer m_outputAlbedoBuffer;
        uint32_t m_width;
        uint32_t m_height;
        uint32_t m_numAccumFrames;

    public:
        Context(bool logging, bool enableRTX, uint32_t maxCallableDepth, uint32_t stackSize, const int32_t* devices, uint32_t numDevices);
        ~Context();

        uint32_t getID() const {
            return m_ID;
        }

        bool RTXEnabled() const {
            return m_RTXEnabled;
        }
        uint32_t getNumDevices() const {
            return m_numDevices;
        }
        int32_t getDeviceIndexAt(uint32_t idx) const {
            if (idx >= m_numDevices)
                return 0xFFFFFFFF;
            return m_devices[idx];
        }

        void bindOutputBuffer(uint32_t width, uint32_t height, uint32_t glBufferID, uint32_t glBufferDenoiseID);
        const void* mapOutputBuffer();
        void unmapOutputBuffer();
		void *mapOutputBufferDenoise();
		void unmapOutputBufferDenoise();
        void getOutputBufferSize(uint32_t* width, uint32_t* height);

        void render(Scene &scene, const Camera* camera, uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames, bool do_denoise=false);
        void debugRender(Scene &scene, const Camera* camera, VLRDebugRenderingMode renderMode, uint32_t shrinkCoeff, bool firstFrame, uint32_t* numAccumFrames);

        const optix::Context &getOptiXContext() const {
            return m_optixContext;
        }

        const optix::Material &getOptiXMaterialDefault() const {
            return m_optixMaterialDefault;
        }
        const optix::Material &getOptiXMaterialWithAlpha() const {
            return m_optixMaterialWithAlpha;
        }

        uint32_t allocateNodeProcedureSet();
        void releaseNodeProcedureSet(uint32_t index);
        void updateNodeProcedureSet(uint32_t index, const Shared::NodeProcedureSet &procSet);

        uint32_t allocateSmallNodeDescriptor();
        void releaseSmallNodeDescriptor(uint32_t index);
        void updateSmallNodeDescriptor(uint32_t index, const Shared::SmallNodeDescriptor &nodeDesc);

        uint32_t allocateMediumNodeDescriptor();
        void releaseMediumNodeDescriptor(uint32_t index);
        void updateMediumNodeDescriptor(uint32_t index, const Shared::MediumNodeDescriptor &nodeDesc);

        uint32_t allocateLargeNodeDescriptor();
        void releaseLargeNodeDescriptor(uint32_t index);
        void updateLargeNodeDescriptor(uint32_t index, const Shared::LargeNodeDescriptor &nodeDesc);

        uint32_t allocateBSDFProcedureSet();
        void releaseBSDFProcedureSet(uint32_t index);
        void updateBSDFProcedureSet(uint32_t index, const Shared::BSDFProcedureSet &procSet);

        uint32_t allocateEDFProcedureSet();
        void releaseEDFProcedureSet(uint32_t index);
        void updateEDFProcedureSet(uint32_t index, const Shared::EDFProcedureSet &procSet);

        const optix::Program &getOptixCallableProgramNullBSDF_setupBSDF() const {
            return m_optixCallableProgramNullBSDF_setupBSDF;
        }
        uint32_t getNullBSDFProcedureSetIndex() const { return m_nullBSDFProcedureSetIndex; }
        const optix::Program &getOptixCallableProgramNullEDF_setupEDF() const {
            return m_optixCallableProgramNullEDF_setupEDF;
        }
        uint32_t getNullEDFProcedureSetIndex() const { return m_nullEDFProcedureSetIndex; }

        uint32_t allocateSurfaceMaterialDescriptor();
        void releaseSurfaceMaterialDescriptor(uint32_t index);
        void updateSurfaceMaterialDescriptor(uint32_t index, const Shared::SurfaceMaterialDescriptor &matDesc);
    };



    class ClassIdentifier {
        ClassIdentifier &operator=(const ClassIdentifier &) = delete;

        const ClassIdentifier* m_baseClass;

    public:
        ClassIdentifier(const ClassIdentifier* baseClass) : m_baseClass(baseClass) {}

        const ClassIdentifier* getBaseClass() const {
            return m_baseClass;
        }
    };



    class TypeAwareClass {
    public:
        virtual const char* getType() const = 0;
        static const ClassIdentifier ClassID;
        virtual const ClassIdentifier& getClass() const { return ClassID; }

        template <class T>
        constexpr bool is() const {
            return &getClass() == &T::ClassID;
        }

        template <class T>
        constexpr bool isMemberOf() const {
            const ClassIdentifier* curClass = &getClass();
            while (curClass) {
                if (curClass == &T::ClassID)
                    return true;
                curClass = curClass->getBaseClass();
            }
            return false;
        }
    };

#define VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE() \
    static const char* TypeName; \
    virtual const char* getType() const { return TypeName; } \
    static const ClassIdentifier ClassID; \
    virtual const ClassIdentifier &getClass() const override { return ClassID; }



    class Object : public TypeAwareClass {
    protected:
        Context &m_context;

    public:
        VLR_DECLARE_TYPE_AWARE_CLASS_INTERFACE();

        Object(Context &context);
        virtual ~Object() {}

        Context &getContext() {
            return m_context;
        }
    };



    // ----------------------------------------------------------------
    // Miscellaneous

    template <typename RealType>
    class DiscreteDistribution1DTemplate {
        optix::Buffer m_PMF;
        optix::Buffer m_CDF;
        RealType m_integral;
        uint32_t m_numValues;

    public:
        void initialize(Context &context, const RealType* values, size_t numValues);
        void finalize(Context &context);

        void getInternalType(Shared::DiscreteDistribution1DTemplate<RealType>* instance) const;
    };

    using DiscreteDistribution1D = DiscreteDistribution1DTemplate<float>;



    template <typename RealType>
    class RegularConstantContinuousDistribution1DTemplate {
        optix::Buffer m_PDF;
        optix::Buffer m_CDF;
        RealType m_integral;
        uint32_t m_numValues;

    public:
        void initialize(Context &context, const RealType* values, size_t numValues);
        void finalize(Context &context);

        RealType getIntegral() const { return m_integral; }
        uint32_t getNumValues() const { return m_numValues; }

        void getInternalType(Shared::RegularConstantContinuousDistribution1DTemplate<RealType>* instance) const;
    };

    using RegularConstantContinuousDistribution1D = RegularConstantContinuousDistribution1DTemplate<float>;



    template <typename RealType>
    class RegularConstantContinuousDistribution2DTemplate {
        optix::Buffer m_raw1DDists;
        RegularConstantContinuousDistribution1DTemplate<RealType>* m_1DDists;
        RegularConstantContinuousDistribution1DTemplate<RealType> m_top1DDist;

    public:
        RegularConstantContinuousDistribution2DTemplate() : m_1DDists(nullptr) {}

        void initialize(Context &context, const RealType* values, size_t numD1, size_t numD2);
        void finalize(Context &context);

        bool isInitialized() const { return m_1DDists != nullptr; }

        void getInternalType(Shared::RegularConstantContinuousDistribution2DTemplate<RealType>* instance) const;
    };

    using RegularConstantContinuousDistribution2D = RegularConstantContinuousDistribution2DTemplate<float>;

    // END: Miscellaneous
    // ----------------------------------------------------------------
}
