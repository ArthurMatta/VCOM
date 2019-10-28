import wx
from gui import GUI

app = wx.App()
gui = GUI(None, title='Barcode Detector')
gui.Show()
app.MainLoop()
