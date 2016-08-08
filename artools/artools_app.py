"""The main artools GUI code."""

#Filename: artools_app.py
#Author: Andrew Nadolski

import os
import wx


class MainWindow(wx.Frame):
    """The main GUI window."""
    def __init__(self, parent, title):
        self.config_dir = ""
        self.config_file = ""
        self.sim_dir = ""
        self.sim_file = ""
        
        wx.Frame.__init__(self, parent, title=title, size=(600,300))
        self.CreateStatusBar()

        # Create the File menu and its buttons
        file_menu = wx.Menu()
        file_about = file_menu.Append(wx.ID_ABOUT, "&About", \
                                          "Information about the program")
        file_open_config = file_menu.Append(wx.ID_ANY, "Open configuration", \
                                                "Open an existing AR configuration")
        file_open_sim = file_menu.Append(wx.ID_OPEN, "Open simulation", \
                                             "Open a previous AR simulation")
        file_exit = file_menu.Append(wx.ID_EXIT, "E&xit", "Exit the program")

        # Make the Menu bar
        menu_bar = wx.MenuBar()
        menu_bar.Append(file_menu, "&File")
        self.SetMenuBar(menu_bar)

        # Make the Event bindings
        self.Bind(wx.EVT_MENU, self.on_about, file_about)
        self.Bind(wx.EVT_MENU, self.on_open_config, file_open_config)
        self.Bind(wx.EVT_MENU, self.on_open_sim, file_open_sim) 
        self.Bind(wx.EVT_MENU, self.on_file_exit, file_exit)

        # Display the window
        self.Show(True)

    def on_about(self, event):
        """Display an 'About program' dialog"""
        content = ("A simple multilayer anti-reflection coating simulator."
                   "\nCopyright, or something, Andrew Nadolski 2016")
        dlg = wx.MessageDialog(self, content, "About artools", wx.OK)
        dlg.ShowModal()
        dlg.Destroy()

    def on_file_exit(self, event):
        """Exit the program"""
        self.Close(True)

    def on_open_config(self, event):
        # NOTE: this doesn't actually open anything yet. It's just a dummy function.
        """Open a previous AR coating configuration file"""
        self.config_dir = ""
        dlg = wx.FileDialog(self, message="Open an existing AR configuration", \
                                defaultDir=self.config_dir, defaultFile="", \
                                wildcard="*")
        if dlg.ShowModal() == wx.ID_OK:
            self.config_file = dlg.GetFilename()
            self.config_dir = dlg.GetDirectory()
            f = open(os.path.join(self.dir_name, self.file_name), 'r')
            self.control.SetValue(f.read())
            f.close()
        dlg.Destroy()
        
    def on_open_sim(self, event):
        """Open a previous artools simulation"""
        dlg = wx.MessageDialog(self, "Open a previous artools simulation", \
                                   "Open previous artools simulation", wx.OK)
        dlg.ShowModal()
        dlg.Destroy()




app = wx.App(False)
frame = MainWindow(None, "artools")
app.MainLoop()
