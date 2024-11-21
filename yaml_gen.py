# This script is used to generate the yaml file for the pruned model.
# This script only fits the yoloface500k model purpose.

import yaml
import torch
import argparse
from models.common import *
from models.yolo import *
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq

module_list = ["Conv", "DWConv", "Bottleneck3", "Concat", "nn.Upsample", "Detect"]

def make_compact_list(data):
    result = CommentedSeq(data)
    result.fa.set_flow_style()  
    return result

def parse_model(model_dict, model):
# parse the model and generate yaml dict

    for i in range(len(model.model)):
        if i < 22:
            part = "backbone"
        else:
            part = "head"
        if isinstance(model.model[i], DWConv):
            model_dict[part].append([-1, 1, "DWConv", [model.model[i].conv.out_channels, model.model[i].conv.kernel_size[0], model.model[i].conv.stride[0]]])
        elif isinstance(model.model[i], Conv):
            model_dict[part].append([-1, 1, "Conv", [model.model[i].conv.out_channels, model.model[i].conv.kernel_size[0], model.model[i].conv.stride[0]]])
        elif isinstance(model.model[i], Bottleneck3):
            model_dict[part].append([-1, 1, "Bottleneck3", [model.model[i].cv3.conv.out_channels, model.model[i].cv1.conv.out_channels]])
        elif isinstance(model.model[i], nn.Sequential):
            for j in range(len(model.model[i])):
                # all Bottleneck3
                model_dict[part].append([-1, 1, "Bottleneck3", [model.model[i][j].cv3.conv.out_channels, model.model[i][j].cv1.conv.out_channels]])
        elif isinstance(model.model[i], Concat):
            if i == 23:
                model_dict[part].append([[-1, -5], 1, "Concat", [1]])
            elif i == 29:
                model_dict[part].append([[-1, 12], 1, "Concat", [1]])
            elif i == 35:
                model_dict[part].append([[-1, 7], 1, "Concat", [1]])
            else:
                # error
                print(f"Error: Concat layer position ({i}) is wrong")
        elif isinstance(model.model[i], nn.Upsample):
            model_dict[part].append([-5, 1, "nn.Upsample", ["None", 2, "nearest"]])
        elif isinstance(model.model[i], Detect):
            model_dict[part].append([[44, 38, 32], 1, "Detect", ["nc", "anchors"]])
        else:
            # error
            print(f"Error: Layer type is not supported: {model.model[i]}")
    # model_dict['backbone'] = make_compact_list(model_dict['backbone'])
    # model_dict['head'] = make_compact_list(model_dict['head'])
    

def dump_yaml(data, file_path):
    class MyDumper(yaml.Dumper):
        def increase_indent(self, flow=False, indentless=False):
            return super(MyDumper, self).increase_indent(flow=flow, indentless=indentless)
    with open(file_path, 'w') as f:
        yaml.dump(data, f, Dumper=MyDumper, default_flow_style=None, sort_keys=False)

def parse_yaml(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    fp = open(file_path.split(".")[0] + "_.yaml", "w")
    
    backbone = False
    head = False
    
    backbone_lines = []
    head_lines = []
    for line in lines:
        if backbone:
            if line.strip() != "head:":
                backbone_lines.append(line)
            else:
                backbone = False
                head = True
            continue
        elif head:
            head_lines.append(line)
            continue
        if line.strip() == "backbone:":
            backbone = True
            continue
        if line.strip() == "head: ":
            head = True
            continue
        if not backbone and not head:
            fp.write(line)
    
    fp.write("\nbackbone:\n")
    fp.write("  [\n")
    component = []
    flag = False
    for idx, line in enumerate(backbone_lines + head_lines):
        if idx == len(backbone_lines) - 1:
            fp.write("  ]\n\nhead:\n  [\n")
        if line.strip().startswith("- -"):
            if line.strip()[4:][0] == '[':
                print()
            if flag:
                string = "    ["
                for i, item in enumerate(component):
                    if i != len(component) - 1:
                        string += str(item) + ", "
                    else:
                        if item.startswith("[None"):
                            # string += "[" + str(item[0]) + ", " + str(item[1]) + '"' + "nearest" + '"' + "]"
                            string += str(item[0:item.find("nearest")]) + ", \"nearest\"]"
                        else:
                            string += str(item)
                fp.write(string + "],\n")
            flag = True
            component = []
            component.append(eval(line.strip()[4:]))
        elif line.strip().startswith("-"):
            if line.strip()[2:] not in module_list:
                component.append(line.strip()[2:])
            else:
                component.append(line.strip()[2:])
                # if line.strip()[2:] == "nn.Upsample":
                #     print()
    string = "    ["
    for i, item in enumerate(component):
        if i != len(component) - 1:
            string += str(item) + ", "
        else:
            string += str(item)
    fp.write(string + "]\n  ]\n")
            
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate yaml file for the pruned model")
    parser.add_argument("--model", type=str, help="Path to the pruned model")

    opt = parser.parse_args()
    model = torch.load(opt.model)['model']          # Load the pruned model
    # print(model)
    # Create the yaml file
    model_dict = {
        "nc": 1,
        "depth_multiple": 1.0,
        "width_multiple": 1.0,
        "anchors": [],
        "backbone": [],
        "head": []
    }
    model_dict['nc'] = 1
    model_dict['depth_multiple'] = 1.0
    model_dict['width_multiple'] = 1.0
    anchors = [[4, 6, 7, 10, 11, 15], [16, 24, 33, 25, 26, 41], [47, 60, 83, 97, 141, 149]]
    # model_dict['anchors'] = make_compact_list(anchors)
    model_dict['anchors'] = anchors
    
    parse_model(model_dict, model)
    
    dump_yaml(model_dict, "pruned_model.yaml")
    parse_yaml("pruned_model.yaml")


