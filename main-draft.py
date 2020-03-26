import pypaper.eval as ppe
import pytoolkit.files as fp
import pypaper.html as pphtml
from easydict import EasyDict as edict

fd = r'F:\text-seg\results\totaltext-ce4\ID0000\output\Epoch000'

subdirs = fp.subdirs(fd)

gt = None
for d in subdirs:
    if 'gt' in d:
        gt = d
        break

for d in subdirs:
    if 'input' in d or 'gt' in d:
        continue
    ppe.ComputeEvalForMethod(d, gt, 3)

HtmlFlags = edict({
    'DisplayReportTime': True,
    'DisplayReportTitle': False,
    'DisplayFileName': False,
    'DisplayFigureCaption': True,
    'DisplayTagVertically': False,
    # ----------------------------------------------------------------------------------------------
    'ReportTitle': '',
    'FontSize': 14,
    'FontFace': 'Microsoft YaHei',
    'ColumnWidth': 320,
    'FigureCaptionFormat': '%7.3f',
    'CaptionDataFilename': 'EvalMatrix.npy',
    'MediaType': 'Image',  # ['Image', 'Video']
    'Transpose': False,
    'NumberOfSamplesPerPage': 0,
    'UseNumAsSampleID': True,  # [True: number, False: alphabet]
    'MaxNumOfSamples': 1e9,
})

t2p = {
    'input':                'input',
    'rs_ou_image_valid':    'rs_ou_image_valid',
    'rs_ou_image_train':    'rs_ou_image_train',
    'gt':                   'gt',
}

hw = pphtml.HtmlWriter(fd, t2p, HtmlFlags)
hw.Run()