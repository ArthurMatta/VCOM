# If this program does not execute properly on your OS because a message like bellow is returned:
#           "This program needs access to the screen. Please run with a
#           Framework build of python, and only when you are logged in
#           on the main display of your Mac."
# Then try doing the following steps:
# 1. Install wxpython globaly by typing "pip3 install wxpython" outside of your virtual environment
# 2. Access the bin folder of your virtual environment: "cd [virtual_env_name]/bin". The [virtual_env_name] folder is
#       usually located on /Users/[username]/.virtualenvs/[virtual_env_name]/
# 3. Replace the python wrapper created by the VirtualEnv with system python:
#       "mv python python_orig"
#       "ln -s /usr/local/bin/python3 python"
# 4. Restart your VirtualEnv: "workon [virtual_env_name]" and run the program, it should work now
# 5. If you're still getting the error message, keep following the tutorial in presented in
#       https://tamarisk.it/running-wxpython-inside-a-python3-virtualenv/

import wx
from barDetection import video_detection, image_detection


class GUI(wx.Frame):
    """
    A Frame to select the desired way to acquire the barcode.
    :return gui:
    """

    def __init__(self, parent, title):
        style = wx.SYSTEM_MENU | wx.MINIMIZE_BOX | wx.CLOSE_BOX
        width, height = 300, 150
        wx.Frame.__init__(self, parent, title=title,
                          size=(width, height), style=style)

        self.CreateStatusBar()
        self.SetStatusText("Welcome to Barcode Detector!")

        st = wx.StaticText(parent=self, label="Barcode Detector",
                           pos=(50, 5), size=(width, height / 10))
        font = st.GetFont()
        font.PointSize += 10
        font = font.Bold()
        st.SetFont(font)

        video_button = wx.Button(
            parent=self, label="Video", name="video", pos=(60, 50), size=(75, 30))
        self.Bind(wx.EVT_BUTTON, self.onclick, video_button)
        image_button = wx.Button(
            parent=self, label="Image", name="image", pos=(165, 50), size=(75, 30))
        self.Bind(wx.EVT_BUTTON, self.onclick, image_button)

    @staticmethod
    def onclick(event):
        """
        Add a click event to a button
        :param event:
        :return:
        """
        button = event.GetEventObject()

        if button.GetName() == 'video':
            video_detection()
        else:
            fd = wx.FileDialog(None, "Select image", wildcard="JPEG files (*.jpg)|*.jpg",
                               style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)

            if fd.ShowModal() == wx.ID_CANCEL:
                return

            pathname = fd.GetPath()
            image_detection(pathname)


if __name__ == '__main__':
    app = wx.App()
    gui = GUI(None, title='Barcode Detector')
    gui.Show()
    app.MainLoop()
