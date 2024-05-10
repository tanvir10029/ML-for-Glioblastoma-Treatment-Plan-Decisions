import wx
import nibabel as nib

import fsleyes.controls.controlpanel as ctrlpanel
import fsleyes.views.viewpanel as viewpanel
import fsleyes.main as fslmain

class OverlayListControl(ctrlpanel.ControlPanel):
    def __init__(self, parent, overlayList, displayCtx, frame):
        super(OverlayListControl, self).__init__(parent, overlayList, displayCtx, frame)

        # Setup the user interface
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.updateBtn = wx.Button(self, label='Update Overlay List and Header')
        self.updateBtn.Bind(wx.EVT_BUTTON, self.onUpdateBtn)
        sizer.Add(self.updateBtn, flag=wx.EXPAND)

        self.overlayListText = wx.TextCtrl(self, style=wx.TE_MULTILINE | wx.TE_READONLY)
        sizer.Add(self.overlayListText, proportion=1, flag=wx.EXPAND)

        self.headerInfoText = wx.TextCtrl(self, style=wx.TE_MULTILINE | wx.TE_READONLY)
        sizer.Add(self.headerInfoText, proportion=1, flag=wx.EXPAND)

        self.SetSizer(sizer)
        self.updateOverlayList()

    def onUpdateBtn(self, event):
        self.updateOverlayList()

    def updateOverlayList(self):
        overlay_names = [overlay.name for overlay in self.overlayList]
        self.overlayListText.SetValue("\n".join(overlay_names))
        self.updateHeaderInfo()

    def updateHeaderInfo(self):
        if self.overlayList:
            selected_overlay = self.overlayList[1]  # Assuming the first overlay is selected
            nifti_file_path = selected_overlay.dataSource

            # Load the NIfTI file using nibabel to access the header
            nifti_img = nib.load(nifti_file_path)
            header_info = str(nifti_img.header)

            # Display the dim field from the header
            dim_info = nifti_img.header.get('descrip')
            self.headerInfoText.SetValue(f"Header Info (dim field): {dim_info}")
