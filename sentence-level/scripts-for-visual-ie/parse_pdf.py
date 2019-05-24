from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTFigure
import os
import json
from multiprocessing import Pool

def parse_text(layout):
    """Function to recursively parse the layout tree."""
    result = []
    if not hasattr(layout, '__iter__'):
        return result
    for lt_obj in layout:
        if isinstance(lt_obj, LTTextLine):
            bbox = lt_obj.bbox
            text = lt_obj.get_text().strip()
            if text != '':
                result += [(bbox, text)]
        else:
            result += parse_text(lt_obj)
    return result


def parse_case(case_path):
    """Parse all the pdf files in the folder."""
    try:
        result = {
            'id': case_path.split('/')[-2], 
            'docs': {}
        }

        for name in os.listdir(case_path):
            if name[0] == '.' or name[-4:] != '.pdf':
                continue
            doc_id = name.split('.')[0]
            result['docs'][doc_id] = {'pages': {}}
            doc_obj = result['docs'][doc_id]

            path = case_path + name
            fp = open(path, 'rb')
            parser = PDFParser(fp)
            doc = PDFDocument(parser)
            rsrcmgr = PDFResourceManager()
            laparams = LAParams(detect_vertical=True, all_texts=True)
            device = PDFPageAggregator(rsrcmgr, laparams=laparams)
            interpreter = PDFPageInterpreter(rsrcmgr, device)

            for page in PDFPage.create_pages(doc):
                interpreter.process_page(page)
                layout = device.get_result()
                doc_obj['pages'][layout.pageid] = {
                    'size': (layout.width, layout.height),
                    'text': parse_text(layout)
                }
                # print(layout.width, layout.height)

        output = open(case_path + 'parsed.json', 'w')
        json.dump(result, output, indent=None)
    except:
        print("Error " + case_path)

    return None



def main(base_path):
    case_list = []
    for direc in os.listdir(base_path):
        path = base_path + direc + '/'
        if not os.path.isdir(path):
            continue
        case_list.append(path)
    # Multiprocessing, for speed up
    pool = Pool(processes=8)
    output = pool.map(parse_case, case_list)


if __name__ == '__main__':
    main('Content/')
