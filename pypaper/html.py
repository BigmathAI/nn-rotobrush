import pytoolkit.files as fp
import os, time, numpy as np
from easydict import EasyDict as edict

class Figure(object):
    def __init__(self, figure_path, caption):
        self.Path = figure_path
        self.Caption = caption

    @property
    def path(self):
        return self.Path

    @property
    def caption(self):
        return self.Caption

class HtmlWriter(object):
    HtmlFlags = edict({
        'DisplayReportTime': False,
        'DisplayReportTitle': False,
        'DisplayFileName': True,
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

    def UpdateConfig(self, HtmlFlags):
        for k, v in HtmlFlags.items():
            self.HtmlFlags[k] = v

    def __init__(self, fd, TitleToPath, HtmlFlags=None):
        if HtmlFlags is not None:
            self.UpdateConfig(HtmlFlags)
        self.fd = fd
        self.TitleToFolder = TitleToPath
        self.FigureMatrix = None

    def GenFigureMatrix(self):
        title_to_fnames = self.GetFilenames(self.HtmlFlags.MediaType)
        title_to_captions = self.GetCaptions()

        M = len(title_to_fnames)
        N = 0 if len(title_to_fnames) == 0 else len(self.GetDictFirstElem(title_to_fnames))
        N = min(N, self.HtmlFlags.MaxNumOfSamples)
        #assert N > 0, 'No Files Found ==> No HTML Generated!'

        self.FigureMatrix = np.zeros([M, N], dtype=Figure)
        for k, (title, fnames) in enumerate(title_to_fnames.items()):
            captions = title_to_captions[title]
            if captions is None:
                self.FigureMatrix[k] = [Figure(fname, None) for fname in fnames[:N]]
            else:
                self.FigureMatrix[k] = [Figure(fname, cap) for fname, cap in zip(fnames[:N], captions[:N])]

        self.FigureMatrix = self.FigureMatrix.transpose() # <<=== MUST BE DONE, NumOfSamples x NumOfMethods
        self.NumOfSamples, self.NumOfMethods = self.FigureMatrix.shape

    def FilterFigures(self):
        pass

    def GenSampleIDs(self, NumOfSamples, UseNumAsSampleID):
        if UseNumAsSampleID == True:
            fmt = ':0{}d'.format(len(str(NumOfSamples)))
            fmt = '{' + fmt + '}'
            rst = [fmt.format(i) for i in range(NumOfSamples)]
            rst = ['(' + x + ')' for x in rst]
        else:
            if NumOfSamples > 26:
                raise ValueError('SampleID Larger than Z')
            rst = ['(%s)' % chr(ord('a') + i) for i in range(NumOfSamples)]
        return rst

    def RenderHtml(self):
        if self.HtmlFlags.Transpose == True:
            self.FigureMatrix = self.FigureMatrix.transpose()
        NR, NC = self.FigureMatrix.shape
        SampleIDs = self.GenSampleIDs(self.NumOfSamples, self.HtmlFlags.UseNumAsSampleID)
        Titles = list(self.TitleToFolder.keys())

        every = self.HtmlFlags.NumberOfSamplesPerPage
        every = self.NumOfSamples if every == 0 else every

        if self.NumOfSamples == 0:
            return

        for k in range(0, self.NumOfSamples, every):
            if every == self.NumOfSamples:
                html_filename = 'index.html'
            else:
                num_pages = self.NumOfSamples // every + 1
                fmt = ':0{}d'.format(len(str(num_pages)))
                fmt = 'index-{' + fmt + '}.html'
                html_filename = fmt.format(k // every)
            html_filename = os.path.join(self.fd, html_filename)

            SubMatrix = None
            SubSampleIDs = None

            if self.HtmlFlags.Transpose == True:
                SubMatrix = self.FigureMatrix[:,k:min(self.NumOfSamples, k+every)]
                SubSampleIDs = SampleIDs[k:min(self.NumOfSamples, k+every)]
            else:
                SubMatrix = self.FigureMatrix[k:min(self.NumOfSamples, k+every)]
                SubSampleIDs = SampleIDs[k:min(self.NumOfSamples, k+every)]

            with open(html_filename, 'w') as f:
                self._write_begin_(f, SubMatrix, SubSampleIDs, list(self.TitleToFolder.keys()), self.HtmlFlags)
                self._write_body_(f, SubMatrix, SubSampleIDs, list(self.TitleToFolder.keys()), self.HtmlFlags)
                self._write_end_(f)

    def Run(self):
        self.GenFigureMatrix()
        self.FilterFigures()
        self.RenderHtml()

    def GetFilenames(self, MediaType):
        def _get_filenames_helper(fd, title2folder, exts):
            filenames = {}
            if len(title2folder) == 0:
                return filenames
            for title, path in title2folder.items():
                fullpath = os.path.join(fd, path)
                fnames = [os.path.join('./', path, fp.stem(x)) for x in fp.dir(fullpath, exts, case_sensitive=True)]
                filenames[title] = fnames
            flens = [len(fnames) for k, fnames in filenames.items()]
            minlen = min(flens)
            filenames = {k: fnames[:minlen] for k, fnames in filenames.items()}
            return filenames
        if MediaType == 'Image':
            return _get_filenames_helper(self.fd, self.TitleToFolder, ['.jpg', '.png', '.gif', '.tiff'])
        elif MediaType == 'Video':
            return _get_filenames_helper(self.fd, self.TitleToFolder, ['.avi', '.mp4'])
        else:
            return {}

    def GetCaptions(self):
        captions = {}
        for title, path in self.TitleToFolder.items():
            npy_filename = os.path.join(self.fd, path, self.HtmlFlags.CaptionDataFilename)
            if os.path.exists(npy_filename):
                caption = np.load(npy_filename, 'r')
                if len(caption.shape) == 1:
                    caption = np.expand_dims(caption, 1)
                elif len(caption.shape) == 3:
                    _, _, c = caption.shape
                    caption = caption[:,:,c // 2]
                elif len(caption.shape) > 3:
                    caption = np.reshape(caption, [caption.shape[0], -1])
            else:
                caption = None
            captions[title] = caption
        return captions

    def GetDictFirstElem(self, x):
        k = list(x.keys())[0]
        return x[k]

    def _write_begin_(self, f, SubMatrix, SubSampleIDs, Titles, HtmlFlags):
        f.write('<html>')
        f.write('<body>')
        f.write('<style>\n'
                'th {font-size: %dpx; font-family: %s}\n' 
                'td {font-size: %dpx; font-family: %s}\n'
                'img {box-shadow: 1px 1px #888; border-radius: 1px; border:1px solid black}\n'
                'p.tag {writing-mode: vertical-lr;transform: rotate(180deg);margin: 0;}\n'
                '</style>' % (HtmlFlags.FontSize,
                              HtmlFlags.FontFace,
                              HtmlFlags.FontSize,
                              HtmlFlags.FontFace))
        f.write('<center><table>\n')
        NR, NC = SubMatrix.shape
        if HtmlFlags.DisplayReportTime == True:
            f.write(
                '<tr><th colspan="%d">Report: %s</th></tr>\n' % (NC + 1, time.strftime('%Y-%m-%d, %H-%M-%S')))
        if HtmlFlags.DisplayReportTitle == True:
            f.write('<tr><th colspan="%d">%s</th></tr>\n' % (NC + 1, self.HtmlFlags.ReportTitle))

        if HtmlFlags.Transpose == True:
            f.write('<tr>\n')
            f.write('<th></th>\n')
            for sample_id in SubSampleIDs:
                f.write('<th>' + sample_id + '</th>\n')
            f.write('</tr>\n')
        else:
            f.write('<tr>\n')
            f.write('<th></th>\n')
            for title in Titles:
                f.write('<th>' + title + '</th>\n')
            f.write('</tr>\n')

    def _write_end_(self, f):
        f.write('</table>\n</center>\n')
        f.write('</body>')
        f.write('</html>')

    def _write_body_(self, f, SubMatrix, SubSampleIDs, Titles, HtmlFlags):
        def write_figure_row(f, tag, paths, HtmlFlags):
            f.write('<tr>\n')

            if HtmlFlags.DisplayTagVertically == True:
                f.write('<td valign="middle" align="right"><p class="tag">'
                        '%s</p></td>' % tag)
            else:
                f.write('<td valign="middle" align="right">%s</td>' % tag)
            for path in paths:
                f.write('\t<td valign="middle" align="center">\n')
                if HtmlFlags.MediaType == 'Video':
                    f.write('\t\t<video width="%d" controls><source src="'
                            + path + '" type="video/mp4"></video>\n\t</td>\n' % HtmlFlags.ColumnWidth)
                else:
                    f.write('\t\t<img src="' + path + '" width="%d"/>\n\t</td>\n' % HtmlFlags.ColumnWidth)
            f.write('</tr>\n')

        def write_filename_row(f, paths, HtmlFlags):
            f.write('<tr>\n')
            f.write('<td></td>')
            for path in paths:
                f.write('<td style="text-align:left;font-family:Consolas;font-size:12px">%s</td>' % fp.stem(path))
            f.write('</tr>\n')

        def write_caption_row(f, captions, HtmlFlags):
            f.write('<tr>\n')
            f.write('<td></td>')
            for cap in captions:
                if cap is None:
                    f.write('\t<td valign="middle" align="center"></td>\n')
                else:
                    vals = cap
                    f.write('\t<td valign="middle" align="center">')
                    for k, v in enumerate(vals):
                        fmt = HtmlFlags.FigureCaptionFormat
                        f.write((fmt if k == 0 else ' | ' + fmt) % v)
                    f.write('</td>\n')
            f.write('</tr>\n')

        for k, RowFigures in enumerate(SubMatrix):
            tag = Titles[k] if HtmlFlags.Transpose == True else SubSampleIDs[k]
            paths = [fig.path for fig in RowFigures]
            captions = [fig.caption for fig in RowFigures]

            if HtmlFlags.DisplayFileName:
                write_filename_row(f, paths, HtmlFlags)
            write_figure_row(f, tag, paths, HtmlFlags)
            if HtmlFlags.DisplayFigureCaption:
                write_caption_row(f, captions, HtmlFlags)


