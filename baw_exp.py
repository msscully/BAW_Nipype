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
from nipype.interfaces.utility import Merge, Split, Function, Rename, IdentityInterface

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
import csv
import sys
import string
import argparse

def get_first_T1_and_T2(in_files,T1_count):
    '''
    Returns the first T1 and T2 file in in_files, based on offset in T1_count.
    '''
    return in_files[0],in_files[T1_count]

def GetExtensionlessBaseName(filename):
    '''
    Get the filename without the extension.  Works for .ext and .ext.gz
    '''
    import os
    currBaseName = os.path.basename(filename)
    currExt = os.path.splitext(currBaseName)[1]
    currBaseName = os.path.splitext(currBaseName)[0]
    if currExt == ".gz":
      currBaseName = os.path.splitext(currBaseName)[0]
      currExt = os.path.splitext(currBaseName)[1]
    return currBaseName

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

def get_list_element( nestedList, index ):
    return nestedList[index]

class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value


def WorkupT1T2(processingLevel,ScanDir, subject_data_file, atlas_fname_wpath, BCD_model_path,
               Version=110, InterpolationMode="Linear", Mode=10,DwiList=[]):
  """
  Run autoworkup on a single subjects data.

  This is the main function to call when processing a single subject worth of
  data.  ScanDir is the base of the directory to place results, T1Images & T2Images
  are the lists of images to be used in the auto-workup. atlas_fname_wpath is
  the path and filename of the atlas to use.
  """

  subjData=csv.reader(open(subject_data_file,'rb'), delimiter=',', quotechar='"')
  myDB=dict()
  multiLevel=AutoVivification()
  for row in subjData:
    currDict=dict()
    if len(row) == 5:
      site=row[0]
      subj=row[1]
      session=row[2]
      T1s=eval(row[3])
      T2s=eval(row[4])
      currDict['T1s']=T1s
      currDict['T2s']=T2s
      currDict['site']=site
      currDict['subj']=subj
      myDB[session]=currDict
      UNIQUE_ID=site+"_"+subj+"_"+session
      multiLevel[UNIQUE_ID]=currDict


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

  ########################################################
  # Run ACPC Detect on first T1 Image - Base Image
  ########################################################
  """TODO: Determine if we want to pass subjectID and scanID, always require full
  paths, get them from the output path, or something else.
  """
  siteSource = pe.Node(interface=IdentityInterface(fields=['sitesubjsession_id']),name='siteSource')
  siteSource.iterables = ('sitesubjsession_id', multiLevel.keys() )

  functScans = 'def sessionImages(db,sitesubjsession_id): return [db[sitesubjsession_id]["T1s"], db[sitesubjsession_id]["T2s"]]'
  nestedImagesListNode =  pe.Node ( interface=Function(input_names=['db','sitesubjsession_id'], output_names=['nestedImageList']), name="nestedImageList")
  nestedImagesListNode.inputs.function_str = functScans
  nestedImagesListNode.inputs.db = multiLevel
  baw200.connect(siteSource,'sitesubjsession_id',nestedImagesListNode,'sitesubjsession_id')

  getImageT1List = pe.Node( Function(function=get_list_element, input_names = ['nestedList', 'index'], output_names = ['T1List']), name="getImageT1List" )
  getImageT1List.inputs.index = 0
  baw200.connect( nestedImagesListNode, 'nestedImageList', getImageT1List, 'nestedList' )

  getImageT2List = pe.Node( Function(function=get_list_element, input_names = ['nestedList', 'index'], output_names = ['T2List']), name="getImageT2List" )
  getImageT2List.inputs.index = 1
  baw200.connect( nestedImagesListNode, 'nestedImageList', getImageT2List, 'nestedList' )

  # selectT1s = pe.Node ( interface=Select(index=[0]), name = "selectT1s" )
  # selectT2s = pe.Node ( interface=Select(index=[1]), name = "selectT2s" )
  # baw200.connect( nestedImagesListNode, "nestedImageList", selectT1s, "inlist" )
  # baw200.connect( nestedImagesListNode, "nestedImageList", selectT2s, "inlist" )

  T1Basename = pe.Node( Function(function=GetExtensionlessBaseName, input_names = ['filename'], output_names = ['T1baseDir']), name="T1Basename")
  baw200.connect( [ (getImageT1List, T1Basename, [(('T1List', get_list_element, 0 ), 'filename')] ), ])

  T2Basename = pe.Node( Function(function=GetExtensionlessBaseName, input_names = ['filename'], output_names = ['T2baseDir']), name="T2Basename")
  baw200.connect( [ (getImageT2List, T2Basename, [(('T2List', get_list_element, 0 ), 'filename')] ), ])

  """SRC_T1_T2 = pe.Node(interface=nio.DataGrabber(outfields=['T2_0_file','T1_0_file']),
                          name='FIRST_T1T2_Raw_Inputs')
  baw200.connect([ (T1Basename, SRC_T1_T2, [(('T1baseDir', os.path.dirname), 'base_directory')]), ])
  SRC_T1_T2.inputs.template = '*'
  SRC_T1_T2.inputs.field_template = dict(T1_0_file=os.path.basename(T1NiftiImageList[0]),
                                    T2_0_file=os.path.basename(T2NiftiImageList[0]))
  SRC_T1_T2.inputs.template_args =  dict(T1_0_file=[[]], T2_0_file=[[]]) #No template args to substitute

  ''' Assumes all T1 and T2 inputs are in the same directory. '''
  T1_file_names = [os.path.basename(f_name) for f_name in T1NiftiImageList]
  getImageT2List.outputs.T2List = [os.path.basename(f_name) for f_name in T2NiftiImageList]
  out_fields = []
  template_args = dict()
  for i,fname in enumerate(T1_file_names):
      out_fields.append("T1_%s"%(i+1))
      template_args["T1_%s"%(i+1)] = [[fname]]
  for i,fname in enumerate(getImageT2List.outputs.T2List):
      out_fields.append("T2_%s"%(i+1))
      template_args["T2_%s"%(i+1)] = [[fname]]

  ALL_SRC_T1_T2 = pe.Node(interface=nio.DataGrabber( outfields=out_fields) , name='T1T2_Raw_Inputs')
  ALL_SRC_T1_T2.inputs.base_directory = os.path.dirname(T1NiftiImageList[0])
  ALL_SRC_T1_T2.inputs.template = '%s'
  # This will match the input file names under the first file names directory.
  ALL_SRC_T1_T2.inputs.template_args = template_args
  """


  ########################################################
  # Run ACPC Detect on First T1 Image
  ########################################################
  BCD = pe.Node(interface=BRAINSConstellationDetector(), name="BCD")
  ##  Use program default BCD.inputs.inputTemplateModel = T1ACPCModelFile
  ##BCD.inputs.outputVolume =   "BCD_OUT" + "_ACPC_InPlace.nii.gz"                #$# T1AcpcImageList
  BCD.inputs.outputResampledVolume = "BCD_OUT" + "_ACPC.nii.gz"
  BCD.inputs.outputTransform =  "BCD_OUT" + "_ACPC_transform.mat"
  BCD.inputs.outputLandmarksInInputSpace = "BCD_OUT" + "_ACPC_Original.fcsv"
  BCD.inputs.outputLandmarksInACPCAlignedSpace = "BCD_OUT" + "_ACPC_Landmarks.fcsv"
  BCD.inputs.outputMRML = "BCD_OUT" + "_ACPC_Scene.mrml"
  BCD.inputs.outputLandmarkWeights = "BCD_OUT" + "_ACPC_Landmarks.wgts"
  BCD.inputs.interpolationMode = InterpolationMode
  BCD.inputs.houghEyeDetectorMode = 1
  BCD.inputs.acLowerBound = 80
  BCD.inputs.llsModel = os.path.join(BCD_model_path,'LLSModel.hdf5')
  BCD.inputs.inputTemplateModel = os.path.join(BCD_model_path,'T1.mdl')

  # Entries below are of the form:
  baw200.connect( [ (getImageT1List, BCD, [(('T1List', get_list_element, 0 ), 'inputVolume')] ), ])
#  baw200.connect([
#                  (SRC_T1_T2,BCD, [('T1_0_file', 'inputVolume')]),
#  ])

  ########################################################
  # Run BROIA to make T1 image as small as possible
  ########################################################
  BROIA = pe.Node(interface=BRAINSROIAuto(), name="BROIA")
  BROIA.inputs.ROIAutoDilateSize = 10
  BROIA.inputs.outputVolumePixelType = "short"
  BROIA.inputs.maskOutput = True
  BROIA.inputs.cropOutput = True
  BROIA.inputs.outputVolume = "BROIA_OUT" + "_ACPC_InPlace_cropped.nii.gz"
  BROIA.inputs.outputROIMaskVolume = "BROIA_OUT" + "_ACPC_InPlace_foreground_seg.nii.gz"

  baw200.connect([
    (BCD,BROIA,[('outputResampledVolume','inputVolume')])
  ])

  if processingLevel > 1:
      ########################################################
      # Run BABC on Multi-modal images
      ########################################################
      #MergeT1T2 = pe.Node(interface=Merge(2),name='MergeT1T2')
      #baw200.connect([ (BROIA,MergeT1T2,[('outputVolume','in1')]) ])
      #baw200.connect( getImageT1List, 'T1List', MergeT1T2, 'in1' )
      #baw200.connect( getImageT2List, 'T2List', MergeT1T2, 'in2' )

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

      def MakeOneFileList(T1List,T2List,altT1):
        full_list=T1List
        full_list.extend(T2List)
        full_list[0]=altT1 # The ACPC ROIcropped T1 replacement image.
        return full_list
      makeImagePathList = pe.Node( Function(function=MakeOneFileList, input_names = ['T1List','T2List','altT1'], output_names = ['imagePathList']), name="makeImagePathList")
      baw200.connect( getImageT1List, 'T1List', makeImagePathList, 'T1List' )
      baw200.connect( getImageT2List, 'T2List', makeImagePathList, 'T2List' )
      baw200.connect( BROIA,    'outputVolume', makeImagePathList, 'altT1' )

      def MakeOneFileTypeList(T1List,T2List):
        input_types =       ["T1"]*len(T1List)
        input_types.extend( ["T2"]*len(T2List) )
        return ",".join(input_types)
      makeImageTypeList = pe.Node( Function(function=MakeOneFileTypeList, input_names = ['T1List','T2List'], output_names = ['imageTypeList']), name="makeImageTypeList")
      baw200.connect( getImageT1List, 'T1List', makeImageTypeList, 'T1List' )
      baw200.connect( getImageT2List, 'T2List', makeImageTypeList, 'T2List' )

      def MakeOutFileList(T1List,T2List):
        def GetExtBaseName(filename):
            '''
            Get the filename without the extension.  Works for .ext and .ext.gz
            '''
            import os
            currBaseName = os.path.basename(filename)
            currExt = os.path.splitext(currBaseName)[1]
            currBaseName = os.path.splitext(currBaseName)[0]
            if currExt == ".gz":
              currBaseName = os.path.splitext(currBaseName)[0]
              currExt = os.path.splitext(currBaseName)[1]
            return currBaseName
        all_files=T1List
        all_files.extend(T2List)
        out_corrected_names=[]
        for i in all_files:
            out_name=GetExtBaseName(i)+"_corrected.nii.gz"
            out_corrected_names.append(out_name)
        return out_corrected_names
      makeOutImageList = pe.Node( Function(function=MakeOutFileList, input_names = ['T1List','T2List'], output_names = ['outImageList']), name="makeOutImageList")
      baw200.connect( getImageT1List, 'T1List', makeOutImageList, 'T1List' )
      baw200.connect( getImageT2List, 'T2List', makeOutImageList, 'T2List' )

      BABC= pe.Node(interface=BRAINSABC(), name="BABC")
      baw200.connect(makeImagePathList,'imagePathList',BABC,'inputVolumes')
      baw200.connect(makeImageTypeList,'imageTypeList',BABC,'inputVolumeTypes')
      baw200.connect(makeOutImageList,'outImageList',BABC,'outputVolumes')
      BABC.inputs.debuglevel = 0
      BABC.inputs.maxIterations = 3
      BABC.inputs.maxBiasDegree = 4
      BABC.inputs.filterIteration = 3
      BABC.inputs.filterMethod = 'GradientAnisotropicDiffusion'
      BABC.inputs.gridSize = [28,20,24]
      BABC.inputs.outputFormat = "NIFTI"
      BABC.inputs.outputLabels = "brain_label_seg.nii.gz"
      BABC.inputs.outputDirtyLabels = "volume_label_seg.nii.gz"
      BABC.inputs.posteriorTemplate = "POSTERIOR_%s.nii.gz"
      BABC.inputs.atlasToSubjectTransform = "atlas_to_subject.mat"
      BABC.inputs.implicitOutputs = ['t1_average_BRAINSABC.nii.gz', 't2_average_BRAINSABC.nii.gz']

      BABC.inputs.resamplerInterpolatorType = InterpolationMode
      ##
      BABC.inputs.outputDir = './'

      baw200.connect(BAtlas,'AtlasPVDefinition_xml',BABC,'atlasDefinition')

      baw200.connect(BLI,'outputTransformFilename',BABC,'atlasToSubjectInitialTransform')
      """
      Get the first T1 and T2 corrected images from BABC
      """
      bfc_files = pe.Node(Function(input_names=['in_files','T1_count'],
                                 output_names=['t1_corrected','t2_corrected'],
                                 function=get_first_T1_and_T2), name='bfc_files')

      bfc_files.inputs.T1_count = len(getImageT1List.outputs.T1List)

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
      """ Commented out as the atlas directory doesn't seem to be setup correctly yet.
      BCM = pe.Node(interface=nio.DataGrabber(out_fields=""), name='BCM')
      BCM.inputs.base_directory = atlas_fname_wpath
      BCM.inputs.template = '%s/%s.%s'
      BCM.inputs.template_args['phi'] = ['','','']
      BCM.inputs.template_args['rho'] = ['','','']
      BCM.inputs.template_args['theta'] = ['','','']
      BCM.inputs.template_args['r_caudate'] = ['','','']
      BCM.inputs.template_args['l_caudate'] = ['','','']
      """


      """
      Create xml file for BRAINSCut
      """
      BRAINSCut_xml_file = ''

      """
      BRAINSCut
      """
      """ COMMENT OUT FOR THE MOMENT
      BRAINSCUT = pe.Node(interface=BRAINSCut(),name="BRAINSCUT")
      BRAINSCUT.inputs.applyModel = True
      BRAINSCUT.inputs.netConfiguration = BRAINSCut_xml_file
      """

      """
      BRAINSTalairach
      Not implemented yet.
      """
      if processingLevel > 2:
          subj_id = os.path.basename(os.path.dirname(os.path.dirname(baw200.base_dir)))
          scan_id = os.path.basename(os.path.dirname(baw200.base_dir))
          """
          Run Freesurfer ReconAll
          """
          reconall = pe.Node(interface=ReconAll(),name="FS510")
          reconall.inputs.subject_id = subj_id+'_'+scan_id
          reconall.inputs.directive = 'all'
          reconall.inputs.subjects_dir = '.'
          baw200.connect(SplitAvgBABC,'avgBABCT1',reconall,'T1_files')

  baw200.run(plugin='MultiProc', plugin_args={'n_procs' : 10})
  #baw200.run()
  #baw200.write_graph()

def main(argv=None):
    if argv == None:
        argv = sys.argv

    # Create and parse input arguments
    parser = argparse.ArgumentParser(description='Runs a mini version of BRAINSAutoWorkup')
    group = parser.add_argument_group('Required')
    group.add_argument('-o', action="store", dest='out_dir', required=True,
                       help='The output directory.')
    group.add_argument('-subject_data', action="store", dest='subject_data_file', required=True,
                       help='A file containing the data to be run through the pipeline')
    group.add_argument('-atlasPath', action="store", dest='atlas_fname_wpath', required=True,
                       help='The path to the Atlas directory used by BRAINSABC.',
                       default='/raid0/homes/hjohnson/src/BRAINS3-Darwin-SuperBuildTest/src/bin/Atlas/Atlas_20110701')
    group.add_argument('-BCDModelPath', action="store", dest='BCD_model_path', required=True,
                       help='The path to the model used by BRAINSConstellationDetector.')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    group.add_argument('-processingLevel', action="store", dest='processingLevel', required=False,
                       help='How much processing should be done, 1=basic, 2= include ABC, 3= include Freesurfer',
                       default=2)
    #parser.add_argument('-v', action='store_false', dest='verbose', default=True,
    #                    help='If not present, prints the locations')
    input_arguments = parser.parse_args()

    OUTDIR=os.path.realpath(input_arguments.out_dir)

    WorkupT1T2(input_arguments.processingLevel,
      OUTDIR,
      input_arguments.subject_data_file,
      input_arguments.atlas_fname_wpath,
      input_arguments.BCD_model_path)

if __name__ == "__main__":
    sys.exit(main())
