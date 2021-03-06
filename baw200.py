#!/usr/bin/python
#
#################################################################################
## Program:   BRAINS (Brain Research: Analysis of Images, Networks, and Systems)
## Module:    $RCSfile: $
## Language:  Python
## Date:      $Date:  $
## Version:   $Revision: $
##
##   Copyright (c) Iowa Mental Health Clinical Research Center. All rights reserved.
##   See BRAINSCopyright.txt or http:/www.psychiatry.uiowa.edu/HTML/Copyright.html
##   for details.
##
##      This software is distributed WITHOUT ANY WARRANTY; without even
##      the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
##      PURPOSE.  See the above copyright notices for more information.
##
#################################################################################
"""Import necessary modules from nipype."""
from nipype.interfaces.base import CommandLine, CommandLineInputSpec, TraitedSpec, File, Directory, traits, isdefined, BaseInterface
from nipype.interfaces.utility import Merge, Split, Function, Rename

import nipype.interfaces.io as nio           # Data i/o
import nipype.pipeline.engine as pe          # pypeline engine

from nipype.interfaces.freesurfer import ReconAll

from nipype.utils.misc import package_check
package_check('numpy', '1.3', 'tutorial1')
package_check('scipy', '0.7', 'tutorial1')
package_check('networkx', '1.0', 'tutorial1')
package_check('IPython', '0.10', 'tutorial1')

from BRAINSConstellationDetector import *
from BRAINSABC import *
from BRAINSDemonWarp import *
from BRAINSFit import *
from BRAINSMush import *
from BRAINSResample import *
from BRAINSROIAuto import *
from BRAINSLandmarkInitializer import *
from BRAINSCut import *
from GradientAnisotropicDiffusionImageFilter import *
from GenerateSummedGradientImage import *

import os
import sys
import string
import argparse

## HACK:  This should be more elegant, and should use a class
##        to encapsulate all this functionality into re-usable
##        components with better documentation.
def rootname(path):
    '''
    Return base filename up to first period.
    '''
    splits = os.path.basename(path).split(".")
    if len(splits) > 1:
        return string.join(splits[:-1], ".")
    else:
        return splits[0]

def extension(path):
    splits = os.path.basename(path).split(".")
    if len(splits) > 1:
        return splits[-1]
    else:
        return ""
  
def get_first_T1_and_T2(in_files,T1_count):
    '''
    Returns the first T1 and T2 file in in_files, based on offset in T1_count.
    '''
    return in_files[0],in_files[T1_count]

def GetExtensionlessBaseName(filename):
    '''
    Get the filename without the extension.  Works for .ext and .ext.gz
    '''
    basename = os.path.basename(filename)
    currExt = extension(basename)
    if currExt == "gz":
        return rootname(rootname(basename))
    else:
        return rootname(basename)

# TODO Remove this function as it is a special case of GetExtensionlessBaseName
def ConstellationBasename(image):
    return os.path.basename(rootname(rootname(image)))

def MakeAtlasNode(atlasDirectory):
    """Gererate a DataGrabber node that creates outputs for all the
    elements of the atlas.
    """
    #Generate by running a file system list "ls -1 $AtlasDir *.nii.gz *.xml *.fcsv *.wgts"
    atlas_file_list="AtlasPVDefinition.xml ALLPVAIR.nii.gz ALLPVBASALTISSUE.nii.gz ALLPVCRBLGM.nii.gz ALLPVCRBLWM.nii.gz ALLPVCSF.nii.gz ALLPVNOTCSF.nii.gz ALLPVNOTGM.nii.gz ALLPVNOTVB.nii.gz ALLPVNOTWM.nii.gz ALLPVSURFGM.nii.gz ALLPVVB.nii.gz ALLPVWM.nii.gz avg_t1.nii.gz avg_t2.nii.gz tempNOTVBBOX.nii.gz template_ABC_lables.nii.gz template_WMPM2_labels.nii.gz template_WMPM2_labels.txt template_brain.nii.gz template_cerebellum.nii.gz template_class.nii.gz template_headregion.nii.gz template_leftHemisphere.nii.gz template_nac_lables.nii.gz template_nac_lables.txt template_rightHemisphere.nii.gz template_t1.nii.gz template_t1_clipped.nii.gz template_t2.nii.gz template_t2_clipped.nii.gz template_ventricles.nii.gz"
    atlas_file_names=atlas_file_list.split(' ')
    atlas_file_names=["AtlasPVDefinition.xml","ALLPVAIR.nii.gz",
                      "ALLPVBASALTISSUE.nii.gz","ALLPVCRBLGM.nii.gz",
                      "ALLPVCRBLWM.nii.gz","ALLPVCSF.nii.gz","ALLPVNOTCSF.nii.gz",
                      "ALLPVNOTGM.nii.gz","ALLPVNOTVB.nii.gz","ALLPVNOTWM.nii.gz",
                      "ALLPVSURFGM.nii.gz","ALLPVVB.nii.gz","ALLPVWM.nii.gz",
                      "avg_t1.nii.gz","avg_t2.nii.gz","tempNOTVBBOX.nii.gz",
                      "template_ABC_lables.nii.gz","template_WMPM2_labels.nii.gz",
                      "template_WMPM2_labels.txt","template_brain.nii.gz",
                      "template_cerebellum.nii.gz","template_class.nii.gz",
                      "template_headregion.nii.gz","template_leftHemisphere.nii.gz",
                      "template_nac_lables.nii.gz","template_nac_lables.txt",
                      "template_rightHemisphere.nii.gz","template_t1.nii.gz",
                      "template_t1_clipped.nii.gz","template_t2.nii.gz",
                      "template_t2_clipped.nii.gz","template_ventricles.nii.gz",
                      "template_landmarks.fcsv","template_landmark_weights.csv",]

    ## Remove filename extensions for images, but replace . with _ for other file types
    atlas_file_keys=[fn.replace('.nii.gz','').replace('.','_') for fn in atlas_file_names]
    
    BAtlas = pe.Node(interface=nio.DataGrabber(outfields=atlas_file_keys),
                                               name='BAtlas')
    BAtlas.inputs.base_directory = atlasDirectory
    BAtlas.inputs.template = '*'
    ## Prefix every filename with atlasDirectory
    atlas_search_paths=['{0}'.format(fn) for fn in atlas_file_names]
    BAtlas.inputs.field_template = dict(zip(atlas_file_keys,atlas_search_paths))
    ## Give 'atlasDirectory' as the substitution argument
    atlas_template_args_match=[ [[]] for i in atlas_file_keys ] ##build a list of proper lenght with repeated entries
    BAtlas.inputs.template_args = dict(zip(atlas_file_keys,atlas_template_args_match))
    return BAtlas
 
def create_BRAINSCut_XML(rho,phi,theta,model,r_probabilityMap,
                         l_probabilityMap,atlasT1,
                         atlasBrain,subjT1,subjT2,subjT1GAD,
                         subjT2GAD,subjSGGAD,subjBrain,
                         atlasToSubj,output_dir):

    import re
    import os
    structure = re.search("r_(\w+)_ProbabilityMap",os.path.basename(r_probabilityMap)).group(1)
    from lxml import etree
    from lxml.builder import E
    xml_output = E("AutoSegProcessDescription",
                   E("DataSet",
                     E("Image",Type="T1",Filename=atlasT1),
                     E("Image",Type="T2",Filename="na"),
                     E("Image",Type="SGGAD",Filename="na"),
                     E("Mask",Type="RegistrationROI",Filename=atlasBrain),
                     E("SpatialLocation",Type="rho",Filename=rho),
                     E("SpatialLocation",Type="phi",Filename=phi),
                     E("SpatialLocation",Type="theta",Filename=theta),
                   Name="template",Type="Atlas"),
                   E("RegistrationConfiguration",ImageTypeToUse="T1",
                     ID="BSpline_ROI",BRAINSROIAutoDilateSize="1"),
                   E("ANNParams",Iterations="100",MaximumVectorsPerEpoch="700000",
                     EpochIterations="100",ErrorInterval="1",DesiredError="0.000001",
                     NumberOfHiddenNodes="30",ActivationSlope="1.0",
                     ActivationMinMax="1.0"),
                   E("NeuralNetParams",MaskSmoothingValue="0.0",
                     GradientProfileSize="1",TrainingModelFilename=model,
                     TrainingVectorFilename="na",TestVectorFilename="na",
                     Normalization="true"),
                   E("ApplyModel",CutOutThresh="0.05",MaskThresh="0.4"),
                   E("ProbabilityMap",StructureID="l_%s"%structure,Gaussian="0.5",
                     GenerateVector="true",Filename=l_probabilityMap),
                   E("ProbabilityMap",StructureID="r_%s"%structure,Gaussian="0.5",
                     GenerateVector="true",Filename=r_probabilityMap),
                   E("Dataset",
                     E("Image",Type="T1",Filename=subjT1),
                     E("Image",Type="T2",Filename=subjT2),
                     E("Image",Type="SGGAD",Filename=subjSGGAD),
                     E("Image",Type="T1GAD",Filename=subjT1GAD),
                     E("Image",Type="T2GAD",Filename=subjT2GAD),
                     E("Image",Type="RegistrationROI",Filename=subjBrain),
                     E("Registration",SubjToAtlasRegistrationFilename="na",
                       AtlasToSubjectRegistrationFilename=atlasToSubj,ID="BSpline_ROI"),
                     Name=structure,Type="Apply",OutputDir=output_dir)
                  )

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    xml_filename = os.path.join(output_dir,'%s.xml' % structure)
    xml_file = open(xml_filename,'w')
    xml_file.write(etree.tostring(xml_output, pretty_print=True))
    return xml_filename


def WorkupT1T2(ScanDir, T1Images, T2Images, atlas_fname_wpath, BCD_model_path,
               run_freesurfer, Version=110, InterpolationMode="Linear", Mode=10,DwiList=[]):
  """ 
  Run autoworkup on a single subjects data.

  This is the main function to call when processing a single subject worth of
  data.  ScanDir is the base of the directory to place results, T1Images & T2Images
  are the lists of images to be used in the auto-workup. atlas_fname_wpath is
  the path and filename of the atlas to use.
  """

  if len(T1Images) < 1:
      raise exception("ERROR:  Length of T1 image list is 0,  at least one T1 image must be specified.")
  if len(T2Images) < 1:
      raise exception("ERROR:  Length of T2 image list is 0,  at least one T2 image must be specified.")

  ########### PIPELINE INITIALIZATION #############
  baw200 = pe.Workflow(name="BAW_200_workflow")
  baw200.config['execution'] = {
                                   'plugin':'Linear',
                                   'stop_on_first_crash':'True',
                                   'stop_on_first_rerun': 'False',
                                   'hash_method': 'timestamp',
                                   'single_thread_matlab':'True',
                                   'remove_unnecessary_outputs':'False',
                                   'use_relative_paths':'False',
                                   'remove_node_directories':'False'
                                   }
  baw200.config['logging'] = {
    'workflow_level':'DEBUG',
    'filemanip_level':'DEBUG',
    'interface_level':'DEBUG'
  }
  baw200.base_dir = ScanDir

  T1NiftiImageList = T1Images
  T2NiftiImageList = T2Images

  T1Basename = GetExtensionlessBaseName(T1NiftiImageList[0])
  T2Basename = GetExtensionlessBaseName(T2NiftiImageList[0])

  ########################################################
  # Run ACPC Detect on first T1 Image - Base Image
  ########################################################
  """TODO: Determine if we want to pass subjectID and scanID, always require full
  paths, get them from the output path, or something else.
  """
  SRC_T1_T2 = pe.Node(interface=nio.DataGrabber(outfields=['T2_0_file','T1_0_file']), 
                          name='FIRST_T1T2_Raw_Inputs')
  SRC_T1_T2.inputs.base_directory = os.path.dirname(T1NiftiImageList[0])
  SRC_T1_T2.inputs.template = '*'
  SRC_T1_T2.inputs.field_template = dict(T1_0_file=os.path.basename(T1NiftiImageList[0]),
                                    T2_0_file=os.path.basename(T2NiftiImageList[0]))
  SRC_T1_T2.inputs.template_args =  dict(T1_0_file=[[]], T2_0_file=[[]]) #No template args to substitute

  ''' Assumes all T1 and T2 inputs are in the same directory. '''
  T1_file_names = [os.path.basename(f_name) for f_name in T1NiftiImageList]
  T2_file_names = [os.path.basename(f_name) for f_name in T2NiftiImageList]
  out_fields = []
  template_args = dict()
  for i,fname in enumerate(T1_file_names):
      out_fields.append("T1_%s"%(i+1))
      template_args["T1_%s"%(i+1)] = [[fname]]
  for i,fname in enumerate(T2_file_names):
      out_fields.append("T2_%s"%(i+1))
      template_args["T2_%s"%(i+1)] = [[fname]]

  ALL_SRC_T1_T2 = pe.Node(interface=nio.DataGrabber( outfields=out_fields) , name='T1T2_Raw_Inputs')
  ALL_SRC_T1_T2.inputs.base_directory = os.path.dirname(T1NiftiImageList[0])
  ALL_SRC_T1_T2.inputs.template = '%s'
  # This will match the input file names under the first file names directory.
  ALL_SRC_T1_T2.inputs.template_args = template_args


  ########################################################
  # Run ACPC Detect on First T1 Image
  ########################################################
  BCD = pe.Node(interface=BRAINSConstellationDetector(), name="BCD")
  ##  Use program default BCD.inputs.inputTemplateModel = T1ACPCModelFile
  ##BCD.inputs.outputVolume =   T1Basename + "_ACPC_InPlace.nii.gz"                #$# T1AcpcImageList
  BCD.inputs.outputResampledVolume = T1Basename + "_ACPC.nii.gz"
  BCD.inputs.outputTransform =  T1Basename + "_ACPC_transform.mat"
  BCD.inputs.outputLandmarksInInputSpace = T1Basename + "_ACPC_Original.fcsv"
  BCD.inputs.outputLandmarksInACPCAlignedSpace = T1Basename + "_ACPC_Landmarks.fcsv"
  BCD.inputs.outputMRML = T1Basename + "_ACPC_Scene.mrml"
  BCD.inputs.outputLandmarkWeights = T1Basename + "_ACPC_Landmarks.wgts"
  BCD.inputs.interpolationMode = InterpolationMode
  BCD.inputs.houghEyeDetectorMode = 1
  BCD.inputs.acLowerBound = 80
  BCD.inputs.llsModel = os.path.join(BCD_model_path,'LLSModel.hdf5')
  BCD.inputs.inputTemplateModel = os.path.join(BCD_model_path,'T1.mdl')
  
  # Entries below are of the form:
  baw200.connect([
                  (SRC_T1_T2,BCD, [('T1_0_file', 'inputVolume')]),
  ])

  ########################################################
  # Run BROIA to make T1 image as small as possible
  ########################################################
  BROIA = pe.Node(interface=BRAINSROIAuto(), name="BROIA")
  BROIA.inputs.ROIAutoDilateSize = 10
  BROIA.inputs.outputVolumePixelType = "short"
  BROIA.inputs.maskOutput = True
  BROIA.inputs.cropOutput = True
  BROIA.inputs.outputVolume = T1Basename + "_ACPC_InPlace_cropped.nii.gz"
  BROIA.inputs.outputROIMaskVolume = T1Basename + "_ACPC_InPlace_foreground_seg.nii.gz"
  
  baw200.connect([
    (BCD,BROIA,[('outputResampledVolume','inputVolume')])
  ])

  ########################################################
  # Run BABC on Multi-modal images
  ########################################################
  inputs_length = len(T1_file_names)+len(T2_file_names)
  MergeT1T2 = pe.Node(interface=Merge(inputs_length),name='MergeT1T2')
  
  BLI = pe.Node(interface=BRAINSLandmarkInitializer(), name="BLI")
  BLI.inputs.outputTransformFilename = "landmarkInitializer_atlas_to_subject_transform.mat"

  BAtlas = MakeAtlasNode(atlas_fname_wpath) ## Call function to create node

  baw200.connect([
      (BCD,BLI,[('outputLandmarksInACPCAlignedSpace','inputFixedLandmarkFilename')]),
  ])
  baw200.connect([
      (BAtlas,BLI,[('template_landmarks_fcsv','inputMovingLandmarkFilename')]),
      (BAtlas,BLI,[('template_landmark_weights_csv','inputWeightFilename')])
  ])

  baw200.connect([
    (BROIA,MergeT1T2,[('outputVolume','in1')])
  ])
  for i,fname in enumerate(T1_file_names[1:]):
      baw200.connect([
          (ALL_SRC_T1_T2,MergeT1T2,[("T1_%s"%(i+2),"in%s"%(i+2))])])
  
  for i,fname in enumerate(T2_file_names):
      baw200.connect([
          (ALL_SRC_T1_T2,MergeT1T2,[("T2_%s"%(i+1),"in%s"%(i+len(T1_file_names)+1))])])
  
  BABC= pe.Node(interface=BRAINSABC(), name="BABC")
  BABC.inputs.debuglevel = 0
  BABC.inputs.maxIterations = 3
  BABC.inputs.maxBiasDegree = 4
  BABC.inputs.filterIteration = 3
  BABC.inputs.filterMethod = 'GradientAnisotropicDiffusion'
  BABC.inputs.gridSize = [28,20,24]
  BABC.inputs.outputFormat = "NIFTI"
  out_vols = []
  input_types = ["T1"]*len(T1_file_names)
  input_types.extend(["T2"]*len(T2_file_names))
  all_t1_t2s = list(T1_file_names)
  all_t1_t2s.extend(T2_file_names)
  for i in all_t1_t2s:
      out_vols.append("%s_corrected%s" %
                      (GetExtensionlessBaseName(i),".nii.gz"))
  BABC.inputs.outputVolumes = out_vols
  BABC.inputs.inputVolumeTypes = input_types
  BABC.inputs.outputLabels = "brain_label_seg.nii.gz"
  BABC.inputs.outputDirtyLabels = "volume_label_seg.nii.gz"
  BABC.inputs.posteriorTemplate = "POSTERIOR_%s.nii.gz"
  BABC.inputs.atlasToSubjectTransform = "atlas_to_subject.mat"
  BABC.inputs.implicitOutputs = ['t1_average_BRAINSABC.nii.gz', 't2_average_BRAINSABC.nii.gz']
  
  BABC.inputs.resamplerInterpolatorType = InterpolationMode
  ##
  BABC.inputs.outputDir = './'
  
  baw200.connect(BAtlas,'AtlasPVDefinition_xml',BABC,'atlasDefinition')
  baw200.connect([
    (MergeT1T2,BABC, [('out','inputVolumes')])
  ])

  baw200.connect(BLI,'outputTransformFilename',BABC,'atlasToSubjectInitialTransform')
  """
  Get the first T1 and T2 corrected images from BABC
  """
  bfc_files = pe.Node(Function(input_names=['in_files','T1_count'],    
                             output_names=['t1_corrected','t2_corrected'], 
                             function=get_first_T1_and_T2), name='bfc_files')
  
  bfc_files.inputs.T1_count = len(T1_file_names)

  baw200.connect(BABC,'outputVolumes',bfc_files,'in_files')
  
  """
  ResampleNACLabels
  """
  ResampleAtlasNACLabels=pe.Node(interface=BRAINSResample(),name="ResampleAtlasNACLabels")
  ResampleAtlasNACLabels.inputs.interpolationMode = "NearestNeighbor"
  ResampleAtlasNACLabels.inputs.outputVolume = "atlasToSubjectNACLabels.nii.gz"
  
  baw200.connect(BABC,'atlasToSubjectTransform',ResampleAtlasNACLabels,'warpTransform')
  baw200.connect(bfc_files,'t1_corrected',ResampleAtlasNACLabels,'referenceVolume')
  baw200.connect(BAtlas,'template_nac_lables',ResampleAtlasNACLabels,'inputVolume')
  
  """
  BRAINSMush
  """
  BMUSH=pe.Node(interface=BRAINSMush(),name="BMUSH")
  BMUSH.inputs.outputVolume = "MushImage.nii.gz"
  BMUSH.inputs.outputMask = "MushMask.nii.gz"
  BMUSH.inputs.lowerThresholdFactor = 1.2
  BMUSH.inputs.upperThresholdFactor = 0.55
  
  baw200.connect(bfc_files,'t1_corrected',BMUSH,'inputFirstVolume')
  baw200.connect(bfc_files,'t2_corrected',BMUSH,'inputSecondVolume')
  baw200.connect(BABC,'outputLabels',BMUSH,'inputMaskVolume')
  
  """
  BRAINSROIAuto
  """
  BROI = pe.Node(interface=BRAINSROIAuto(), name="BRAINSROIAuto")
  BROI.inputs.closingSize=12
  BROI.inputs.otsuPercentileThreshold=0.01
  BROI.inputs.thresholdCorrectionFactor=1.0
  BROI.inputs.outputROIMaskVolume = "temproiAuto_t1_ACPC_corrected_BRAINSABC.nii.gz"
  baw200.connect(bfc_files,'t1_corrected',BROI,'inputVolume')

  """
  Split the implicit outputs of BABC
  """
  SplitAvgBABC = pe.Node(Function(input_names=['in_files','T1_count'], output_names=['avgBABCT1','avgBABCT2'], 
                           function = get_first_T1_and_T2), name="SplitAvgBABC")
  SplitAvgBABC.inputs.T1_count = 1 ## There is only 1 average T1 image.

  baw200.connect(BABC,'implicitOutputs',SplitAvgBABC,'in_files')


  """
  Gradient Anistropic Diffusion images for BRAINSCut
  """
  GADT1=pe.Node(interface=GradientAnisotropicDiffusionImageFilter(),name="GADT1")
  GADT1.inputs.timeStep = 0.05
  GADT1.inputs.conductance = 1
  GADT1.inputs.numberOfIterations = 5
  GADT1.inputs.outputVolume = "GADT1.nii.gz"

  baw200.connect(SplitAvgBABC,'avgBABCT1',GADT1,'inputVolume')

  GADT2=pe.Node(interface=GradientAnisotropicDiffusionImageFilter(),name="GADT2")
  GADT2.inputs.timeStep = 0.05
  GADT2.inputs.conductance = 1
  GADT2.inputs.numberOfIterations = 5
  GADT2.inputs.outputVolume = "GADT2.nii.gz"

  baw200.connect(SplitAvgBABC,'avgBABCT2',GADT2,'inputVolume')

  """
  Sum the gradient images for BRAINSCut
  """
  SGI=pe.Node(interface=GenerateSummedGradientImage(),name="SGI")
  SGI.inputs.outputFileName = "SummedGradImage.nii.gz"

  baw200.connect(GADT1,'outputVolume',SGI,'inputVolume1')
  baw200.connect(GADT2,'outputVolume',SGI,'inputVolume2')


  """
  Load the BRAINSCut models & probabiity maps.
  """
  BRAINSCut_structures = ['caudate','thalamus','putamen','hippocampus']
  BCM_outputs = ['phi','rho','theta','r_probabilityMaps',
                 'l_probabilityMaps','models']

  BCM_Models = pe.Node(interface=nio.DataGrabber(infields=['structures'],
                                                 outfields=BCM_outputs),
                       name='BCM_Models')
  BCM_Models.inputs.base_directory = atlas_fname_wpath
  BCM_Models.inputs.template = '%s/%s.%s'
  BCM_Models.inputs.structures = BRAINSCut_structures
  BCM_Models.inputs.template_args['phi'] = [['spatialImages','phi','nii.gz']]
  BCM_Models.inputs.template_args['rho'] = [['spatialImages','rho','nii.gz']]
  BCM_Models.inputs.template_args['theta'] = [['spatialImages','theta','nii.gz']]
  BCM_Models.inputs.field_template = dict(
      r_probabilityMaps='probabilityMaps/r_%s_ProbabilityMap.nii.gz',
      l_probabilityMaps='probabilityMaps/l_%s_ProbabilityMap.nii.gz',
      models='modelFiles/%sModel*',
      )
  BCM_Models.inputs.template_args['r_probabilityMaps'] = [['structures']]
  BCM_Models.inputs.template_args['l_probabilityMaps'] = [['structures']]
  BCM_Models.inputs.template_args['models'] = [['structures']]

  """
  The xml creation and BRAINSCut need to be their own mini-pipeline that gets
  executed once for each of the structures in BRAINSCut_structures.  This can be
  accomplished with a map node and a new pipeline.
  """

  """
  Create xml file for BRAINSCut
  """

  CreateBRAINSCutXML = pe.MapNode(Function(input_names=['rho','phi','theta',
                                                        'model',
                                                        'r_probabilityMap',
                                                        'l_probabilityMap',
                                                        'atlasT1','atlasBrain',
                                                        'subjT1','subjT2',
                                                        'subjT1GAD','subjT2GAD',
                                                        'subjSGGAD','subjBrain',
                                                        'atlasToSubj','output_dir'],
                                           output_names=['xml_filename'], 
                                           function = create_BRAINSCut_XML),
                                  iterfield=['r_probabilityMap',
                                             'l_probabilityMap','model'],
                                  overwrite = True,
                                  name="CreateBRAINSCutXML")

  CreateBRAINSCutXML.inputs.output_dir = os.path.join(baw200.base_dir,
                                                      "BRAINSCut_output")
  baw200.connect(BCM_Models,'rho',CreateBRAINSCutXML,'rho')
  baw200.connect(BCM_Models,'phi',CreateBRAINSCutXML,'phi')
  baw200.connect(BCM_Models,'theta',CreateBRAINSCutXML,'theta')
  baw200.connect(BCM_Models,'models',CreateBRAINSCutXML,'model')
  baw200.connect(BCM_Models,'r_probabilityMaps',CreateBRAINSCutXML,'r_probabilityMap')
  baw200.connect(BCM_Models,'l_probabilityMaps',CreateBRAINSCutXML,'l_probabilityMap')
  baw200.connect(BAtlas,'template_t1',CreateBRAINSCutXML,'atlasT1')
  baw200.connect(BAtlas,'template_brain',CreateBRAINSCutXML,'atlasBrain')
  baw200.connect(SplitAvgBABC,'avgBABCT1',CreateBRAINSCutXML,'subjT1')
  baw200.connect(SplitAvgBABC,'avgBABCT2',CreateBRAINSCutXML,'subjT2')
  baw200.connect(GADT1,'outputVolume',CreateBRAINSCutXML,'subjT1GAD')
  baw200.connect(GADT2,'outputVolume',CreateBRAINSCutXML,'subjT2GAD')
  baw200.connect(SGI,'outputFileName',CreateBRAINSCutXML,'subjSGGAD')
  baw200.connect(BABC,'outputLabels',CreateBRAINSCutXML,'subjBrain')
  baw200.connect(BABC,'atlasToSubjectTransform',CreateBRAINSCutXML,'atlasToSubj')

  """
  BRAINSCut
  """
  BRAINSCUT = pe.MapNode(interface=BRAINSCut(),name="BRAINSCUT",
                         iterfield=['netConfiguration'])
  BRAINSCUT.inputs.applyModel = True
  baw200.connect(CreateBRAINSCutXML,'xml_filename',BRAINSCUT,'netConfiguration')

  """
  BRAINSTalairach
  Not implemented yet.
  """

  subj_id = os.path.basename(os.path.dirname(os.path.dirname(baw200.base_dir)))
  scan_id = os.path.basename(os.path.dirname(baw200.base_dir))
  """
  Run Freesurfer ReconAll
  """
  if run_freesurfer:
      reconall = pe.Node(interface=ReconAll(),name="FS510")
      reconall.inputs.subject_id = subj_id+'_'+scan_id
      reconall.inputs.directive = 'all'
      reconall.inputs.subjects_dir = '.'
      baw200.connect(SplitAvgBABC,'avgBABCT1',reconall,'T1_files')
  else:
      print "Skipping freesurfer"

#  reconall.cmdline
  
  baw200.run()
  #baw200.write_graph()
  
def main(argv=None):
    if argv == None:
        argv = sys.argv

    # Create and parse input arguments
    parser = argparse.ArgumentParser(description='Runs a mini version of BRAINSAutoWorkup')
    group = parser.add_argument_group('Required')
    group.add_argument('-o', action="store", dest='out_dir', required=True,
                       help='The output directory.')
    group.add_argument('-t1s', action="store", dest='t1', required=True,
                       help='A comma seperated list of T1 images. At least one image required.')
    group.add_argument('-t2s', action="store", dest='t2', required=True,
                       help='A comma seperated list of T2 images. At least one image required.')
    group.add_argument('-atlasPath', action="store", dest='atlas_fname_wpath', required=True,
                       help='The path to the Atlas directory used by BRAINSABC.',
                       default='/raid0/homes/hjohnson/src/BRAINS3-Darwin-SuperBuildTest/src/bin/Atlas/Atlas_20110701')
    group.add_argument('-BCDModelPath', action="store", dest='BCD_model_path', required=True,
                       help='The path to the model used by BRAINSConstellationDetector.')
    group.add_argument('-runFreesurfer', action="store_true", dest='run_freesurfer', default=False,
                       help='Should the freesurfer pipeline be run?.')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    #parser.add_argument('-v', action='store_false', dest='verbose', default=True,
    #                    help='If not present, prints the locations')
    input_arguments = parser.parse_args()

    OUTDIR=os.path.realpath(input_arguments.out_dir)

    if len(input_arguments.t1) < 1:
        print "ERROR:  Length of T1 image list is 0,  at least one T1 image must be specified."
        sys.exit(-1)
    if len(input_arguments.t2) < 1:
        print "ERROR:  Length of T2 image list is 0,  at least one T2 image must be specified."
        sys.exit(-1)

    WorkupT1T2(OUTDIR,input_arguments.t1.split(','),input_arguments.t2.split(','),input_arguments.atlas_fname_wpath,
              input_arguments.BCD_model_path,input_arguments.run_freesurfer)

if __name__ == "__main__":
    sys.exit(main())
