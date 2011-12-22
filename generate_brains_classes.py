from nipype.interfaces.slicer import generate_classes as SEM
import subprocess
import argparse

if __name__ == "__main__":
    # Create and parse input arguments
    parser = argparse.ArgumentParser(description='Runs a mini version of BRAINSAutoWorkup')
    group = parser.add_argument_group('Required')
    group.add_argument('-p', action="store", dest='exec_dir', required=True,
                       help='SEM programs path.',
                       default='/hjohnson/NAMIC/msscully/BRAINSia/Slicer-itkv4/Slicer-build/plugins')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    input_arguments = parser.parse_args()

    # removed BRAINSEyeDetector because it doesn't define channels
    test_list="""    
BRAINSABC
BRAINSAlignMSP
BRAINSClipInferior
BRAINSConstellationDetector
BRAINSConstellationModeler
BRAINSLandmarkInitializer
BRAINSDemonWarp
BRAINSFit
BRAINSMush
BRAINSInitializedControlPoints
BRAINSLinearModelerEPCA
BRAINSLmkTransform
BRAINSMultiModeSegment
BRAINSROIAuto
BRAINSResample
BRAINSTrimForegroundInDirection
ESLR
GenerateLabelMapFromProbabilityMap
VBRAINSDemonWarp
extractNrrdVectorIndex
gtractAnisotropyMap
gtractAverageBvalues
gtractClipAnisotropy
gtractCoRegAnatomy
gtractConcatDwi
gtractCopyImageOrientation
gtractCoregBvalues
gtractCostFastMarching
gtractImageConformity
gtractInvertBSplineTransform
gtractInvertDeformationField
gtractInvertRigidTransform
gtractResampleAnisotropy
gtractResampleB0
gtractResampleCodeImage
gtractResampleDWIInPlace
gtractTensor
gtractTransformToDeformationField
GradientAnisotropicDiffusionImageFilter
GenerateSummedGradientImage
"""
    SEM_exe=list()
    for candidate_exe in test_list.split():
        test_command=input_arguments.exec_dir+"/"+candidate_exe+" --xml"
        xmlReturnValue = subprocess.Popen(test_command, stdout=subprocess.PIPE, shell=True).communicate()[0]
        isThisAnSEMExecutable = xmlReturnValue.find("<executable>")
        if isThisAnSEMExecutable != -1:
            SEM_exe.append(candidate_exe)

    print("SEM_PROGRAMS: {0}".format(SEM_exe))
    
    ## NOTE:  For now either the launcher needs to be found on the default path, or
    ##        every tool in the modules list must be found on the default path
    ##        AND calling the module with --xml must be supported and compliant.
    ##       modules_list = ['BRAINSConstellationDetector','BRAINSFit', 'BRAINSResample', 'BRAINSDemonWarp', 'BRAINSROIAuto']
    ## SlicerExecutionModel compliant tools that are usually statically built, and don't need the Slicer3 --launcher
    SEM.generate_all_classes(modules_list=SEM_exe,launcher=[])


### This uses the unsuppored "point" SEM type
### TransformFromFiducials
### This uses the unsupported "geometry" SEM type
### compareTractInclusion
### gtractCreateGuideFiber
### gtractFiberTracking
### gtractCostFastMarching
### gtractFastMarchingTracking
### gtractResampleFibers
