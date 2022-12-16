from typing import List
import xml.etree.cElementTree as ET
from xml.dom import minidom
import shutil
import os


def output_XML(id: List, text: List, labels: List):
    root_element = ET.Element("reviews")
    for i in range(len(id)):
        subb = ET.SubElement(root_element, "review", attrib={"id": str(id[i]), "polarity": str(labels[i])})
        subb.text = text[i]

    xml_string = ET.tostring(root_element)
    dom = minidom.parseString(xml_string)
    with open(r"output.xml_old", 'r+', encoding='utf-8') as f:
        dom.writexml(f, indent='', newl='\n',
                     addindent='\t', encoding='utf-8')

        f.seek(0)
        f.readline()
        target_file = open('output.xml', 'w')
        shutil.copyfileobj(f, target_file)
        target_file.close()

    os.remove("output.xml_old")


# output_XML(list(range(3)), ["hello", "good", "perfect"], [1, 0, 1])