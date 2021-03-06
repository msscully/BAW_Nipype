import xml.dom.minidom
import subprocess
import os
import warnings
from nipype.interfaces.base import (CommandLineInputSpec, CommandLine, traits,
                                    TraitedSpec, File, StdOutCommandLine,
                                    StdOutCommandLineInputSpec, isdefined)


def generate_all_classes(modules_list = [], launcher=[]):
    """ modules_list contains all the SEM compliant tools that should have wrappers created for them.
        launcher containtains the command line prefix wrapper arugments needed to prepare
        a proper environment for each of the modules.
    """
    init_imports = ""
    for module in modules_list:
        module_python_filename="%s.py"%module
        print("="*80)
        print("Generating Definition for module {0} in {1}".format(module,module_python_filename))
        print("^"*80)
        code = generate_class(module,launcher)
        f = open(module_python_filename, "w")
        f.write(code)
        f.close()
        init_imports += "from %s import %s\n"%(module,module)
    f = open("__init__.py", "w")
    f.write(init_imports)
    f.close()


def generate_class(module,launcher):
    dom = _grab_xml(module,launcher)
    inputTraits = []
    outputTraits = []
    outputs_filenames = {}

    #self._outputs_nodes = []

    for paramGroup in dom.getElementsByTagName("parameters"):
        for param in paramGroup.childNodes:
            if param.nodeName in ['label', 'description', '#text', '#comment']:
                continue
            traitsParams = {}

            name = param.getElementsByTagName('name')[0].firstChild.nodeValue
            name = name.lstrip().rstrip()

            longFlagNode = param.getElementsByTagName('longflag')
            if longFlagNode:
                ## Prefer to use longFlag as name if it is given, rather than the parameter name
                longFlagName = longFlagNode[0].firstChild.nodeValue
                ## SEM automatically strips prefixed "--" or "-" from from xml before processing
                ##     we need to replicate that behavior here The following
                ##     two nodes in xml have the same behavior in the program
                ##     <longflag>--test</longflag>
                ##     <longflag>test</longflag>
                longFlagName = longFlagName.lstrip(" -").rstrip(" ")
                traitsParams["argstr"] = "--" + longFlagName + " "
            else:
                traitsParams["argstr"] = "--" + name + " "

            argsDict = {'directory': '%s', 'file': '%s', 'integer': "%d", 'double': "%f", 'float': "%f", 'image': "%s", 'transform': "%s", 'boolean': '', 'string-enumeration': '%s', 'string': "%s", 'integer-enumeration' : '%s'}

            if param.nodeName.endswith('-vector'):
                traitsParams["argstr"] += "%s"
            else:
                traitsParams["argstr"] += argsDict[param.nodeName]

            index = param.getElementsByTagName('index')
            if index:
                traitsParams["position"] = index[0].firstChild.nodeValue

            desc = param.getElementsByTagName('description')
            if index:
                traitsParams["desc"] = desc[0].firstChild.nodeValue

            name = param.getElementsByTagName('name')[0].firstChild.nodeValue

            typesDict = {'integer': "traits.Int", 'double': "traits.Float",
                         'float': "traits.Float", 'image': "File",
                         'transform': "File", 'boolean': "traits.Bool",
                         'string': "traits.Str", 'file':"File",
                         'directory': "Directory"}

            if param.nodeName.endswith('-enumeration'):
                type = "traits.Enum"
                values = ['"%s"'%el.firstChild.nodeValue for el in param.getElementsByTagName('element')]
            elif param.nodeName.endswith('-vector'):
                type = "InputMultiPath"
                if param.nodeName in ['file', 'directory', 'image', 'transform']:
                    values = ["%s(exists=True)"%typesDict[param.nodeName.replace('-vector','')]]
                else:
                    values = [typesDict[param.nodeName.replace('-vector','')]]
                traitsParams["sep"] = ','
            elif param.getAttribute('multiple') == "true":
                type = "InputMultiPath"
                if param.nodeName in ['file', 'directory', 'image', 'transform']:
                    values = ["%s(exists=True)"%typesDict[param.nodeName]]
                else:
                    values = [typesDict[param.nodeName]]
                traitsParams["argstr"] += "..."
            else:
                values = []
                type = typesDict[param.nodeName]

            if param.nodeName in ['file', 'directory', 'image', 'transform'] and param.getElementsByTagName('channel')[0].firstChild.nodeValue == 'output':
                traitsParams["hash_files"] = False
                inputTraits.append("%s = traits.Either(traits.Bool, %s(%s), %s)"%(name,
                                                                                 type,
                                                                                 _parse_values(values).replace("exists=True",""),
                                                                                 _parse_params(traitsParams)))
                traitsParams["exists"] = True
                traitsParams.pop("argstr")
                traitsParams.pop("hash_files")
                outputTraits.append("%s = %s(%s %s)"%(name, type.replace("Input", "Output"), _parse_values(values), _parse_params(traitsParams)))

                outputs_filenames[name] = gen_filename_from_param(param)
            else:
                if param.nodeName in ['file', 'directory', 'image', 'transform'] and type not in ["InputMultiPath", "traits.List"]:
                    traitsParams["exists"] = True

                inputTraits.append("%s = %s(%s %s)"%(name, type, _parse_values(values), _parse_params(traitsParams)))

    input_spec_code = "class " + module + "InputSpec(CommandLineInputSpec):\n"
    for trait in inputTraits:
        input_spec_code += "    " + trait + "\n"

    output_spec_code = "class " + module + "OutputSpec(TraitedSpec):\n"
    if len(outputTraits) > 0:
        for trait in outputTraits:
            output_spec_code += "    " + trait + "\n"
    else:
        output_spec_code +=  "    pass\n"

    output_filenames_code = "_outputs_filenames = {"
    output_filenames_code += ",".join(["'%s':'%s'"%(key,value) for key,value in outputs_filenames.iteritems()])
    output_filenames_code += "}"


    input_spec_code += "\n\n"
    output_spec_code += "\n\n"

    imports = """from nipype.interfaces.base import CommandLine, CommandLineInputSpec, TraitedSpec, File, Directory, traits, isdefined, InputMultiPath, OutputMultiPath
import os\n\n"""

    template = """class %name%(CommandLine):

    input_spec = %name%InputSpec
    output_spec = %name%OutputSpec
    _cmd = "%launcher% %name% "
    %output_filenames_code%

    def _list_outputs(self):
        outputs = self.output_spec().get()
        for name in outputs.keys():
            coresponding_input = getattr(self.inputs, name)
            if isdefined(coresponding_input):
                if isinstance(coresponding_input, bool) and coresponding_input == True:
                    outputs[name] = os.path.abspath(self._outputs_filenames[name])
                else:
                    if isinstance(coresponding_input, list):
                        outputs[name] = [os.path.abspath(inp) for inp in coresponding_input]
                    else:
                        outputs[name] = os.path.abspath(coresponding_input)
        return outputs

    def _format_arg(self, name, spec, value):
        if name in self._outputs_filenames.keys():
            if isinstance(value, bool):
                if value == True:
                    value = os.path.abspath(self._outputs_filenames[name])
                else:
                    return ""
        return super(%name%, self)._format_arg(name, spec, value)\n\n"""


    main_class = template.replace("%name%", module).replace("%output_filenames_code%", output_filenames_code).replace("%launcher%"," ".join(launcher))

    return imports + input_spec_code + output_spec_code + main_class

def _grab_xml(module,launcher):
#        cmd = CommandLine(command = "Slicer3", args="--launch %s --xml"%module)
#        ret = cmd.run()
        command_list=launcher[:] ## force copy to preserve original
        command_list.extend([module, "--xml"])
        final_command=" ".join(command_list)
        xmlReturnValue = subprocess.Popen(final_command, stdout=subprocess.PIPE, shell=True).communicate()[0]
        return xml.dom.minidom.parseString(xmlReturnValue)
#        if ret.runtime.returncode == 0:
#            return xml.dom.minidom.parseString(ret.runtime.stdout)
#        else:
#            raise Exception(cmd.cmdline + " failed:\n%s"%ret.runtime.stderr)
def _parse_params(params):
    list = []
    for key, value in params.iteritems():
        if isinstance(value, str) or isinstance(value, unicode):
            list.append('%s = "%s"'%(key, value))
        else:
            list.append('%s = %s'%(key, value))

    return ",".join(list)

def _parse_values(values):
    values = ['%s'%value for value in values]
    if len(values) > 0:
        retstr = ",".join(values) + ","
    else:
        retstr = ""
    return retstr

def gen_filename_from_param(param):
    base = param.getElementsByTagName('name')[0].firstChild.nodeValue
    fileExtensions = param.getAttribute("fileExtensions")
    if fileExtensions:
        ## It is possible that multiple file extensions can be specified in a
        ## comma separated list,  This will extract just the first extension
        firstFileExtension=fileExtensions.split(',')[0]
        ext = firstFileExtension
    else:
        ext = {'image': '.nii', 'transform': '.mat', 'file': '', 'directory': ''}[param.nodeName]
    return base + ext

if __name__ == "__main__":
    ## NOTE:  For now either the launcher needs to be found on the default path, or
    ##        every tool in the modules list must be found on the default path
    ##        AND calling the module with --xml must be supported and compliant.
    modules_list = ['BRAINSFit', 'BRAINSResample', 'BRAINSDemonWarp', 'BRAINSROIAuto']
    ## SlicerExecutionModel compliant tools that are usually statically built, and don't need the Slicer3 --launcher
    #generate_all_classes(modules_list=modules_list,launcher=[])
    ## Tools compliant with SlicerExecutionModel called from the Slicer environment (for shared lib compatibility)
    launcher=['Slicer3','--launch']
    generate_all_classes(modules_list=modules_list, launcher=launcher )
    #generate_all_classes(modules_list=['BRAINSABC'], launcher=[] )

