"""The main artools GUI code."""

#Filename: artools_app.py
#Author: Andrew Nadolski

import os
import wx
import wx.lib.mixins.listctrl as listmix
import materials as mats


class MainWindow(wx.Frame):
    """The main GUI window."""
    def __init__(self, parent, title):
        self.config_dir = ""
        self.config_file = ""
        self.sim_dir = ""
        self.sim_file = ""
        
        wx.Frame.__init__(self, parent, title=title, size=(800,600))
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

        # Create the OHSHIT menu and its buttons
        ohshit_menu = wx.Menu()
        ohshit_reset_mats = ohshit_menu.Append(wx.ID_ANY, "Reset materials list", \
                                                  "Restore default AR material list")

        # Make the Menu bar
        menu_bar = wx.MenuBar()
        menu_bar.Append(file_menu, "&File")
        menu_bar.Append(ohshit_menu, "OH SHIT")
        self.SetMenuBar(menu_bar)

        # Create the material list panel
        self.ar_list_panel = ARListCtrlPanel(self)
        
        # Make the main window event bindings
        self.Bind(wx.EVT_MENU, self.on_about, file_about)
        self.Bind(wx.EVT_MENU, self.on_open_config, file_open_config)
        self.Bind(wx.EVT_MENU, self.on_open_sim, file_open_sim) 
        self.Bind(wx.EVT_MENU, self.on_file_exit, file_exit)
        self.Bind(wx.EVT_MENU, self.on_reset_mats, ohshit_reset_mats)

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

    def on_reset_mats(self, event):
        warning = ("Are you absolutely sure you want to reset the materials list?"
                   "\n\nThere's no turning back if you click 'Yes'.")
        caption = "WHAT ARE YOU DOING?"
        dlg = wx.MessageDialog(self, warning, caption, wx.YES_NO)
        if dlg.ShowModal() == wx.ID_YES:
            # Clear existing list
            self.ar_list_panel.ar_list.ClearAll()
            # Reset column headers
            self.ar_list_panel.ar_list.InsertColumn(col=0, heading="Material", \
                                                        format=wx.LIST_FORMAT_LEFT)
            self.ar_list_panel.ar_list.InsertColumn(col=1, \
                                                        heading="Dielectric constant",\
                                                        format=wx.LIST_FORMAT_CENTER)
            self.ar_list_panel.ar_list.InsertColumn(col=2, heading="Loss tangent",\
                                                        format=wx.LIST_FORMAT_RIGHT)

            # Repopulate the list with items from 'materials.py'
            sorted_mats = sorted(mats.Electrical.DIELECTRIC.items())
            index = 0
            for material, val in sorted_mats:
                dielectric = str(mats.Electrical.DIELECTRIC[material])
                loss = str(mats.Electrical.LOSS_TAN[material])
                self.ar_list_panel.ar_list.InsertStringItem(index=index, \
                                                                label=material)
                self.ar_list_panel.ar_list.SetStringItem(index=index, col=1, \
                                                             label=dielectric)
                self.ar_list_panel.ar_list.SetStringItem(index=index, col=2, \
                                                             label=loss)
                index += 1
                self.ar_list_panel.ar_list.SetColumnWidth(col=0, \
                                                              width=wx.LIST_AUTOSIZE)
                self.ar_list_panel.ar_list.SetColumnWidth(col=1, \
                                                              width=wx.LIST_AUTOSIZE)
                self.ar_list_panel.ar_list.SetColumnWidth(col=2, \
                                                              width=wx.LIST_AUTOSIZE)
        event.Skip()

class ARListCtrl(wx.ListCtrl):
    def __init__(self, parent, ID=wx.ID_ANY, pos=wx.DefaultPosition, \
                     size=wx.DefaultSize, style=0):
        wx.ListCtrl.__init__(self, parent, ID, pos, size, style)


class ARListCtrlPanel(wx.Panel, listmix.ColumnSorterMixin):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1, style=wx.WANTS_CHARS | wx.SUNKEN_BORDER)
        
        # Make a control list box and name the columns
        self.ar_list = ARListCtrl(self, size=wx.DefaultSize, style=wx.LC_REPORT)
        self.ar_list.InsertColumn(col=0, heading="Material", \
                                      format=wx.LIST_FORMAT_LEFT)
        self.ar_list.InsertColumn(col=1, heading="Dielectric constant",\
                                      format=wx.LIST_FORMAT_CENTER)
        self.ar_list.InsertColumn(col=2, heading="Loss tangent",\
                                      format=wx.LIST_FORMAT_RIGHT)
        
        # Populate the list with items from 'materials.py'
        sorted_mats = sorted(mats.Electrical.DIELECTRIC.items())
        index = 0
        for material, val in sorted_mats:
            dielectric = str(mats.Electrical.DIELECTRIC[material])
            loss = str(mats.Electrical.LOSS_TAN[material])
            self.ar_list.InsertStringItem(index=index, label=material)
            self.ar_list.SetStringItem(index=index, col=1, label=dielectric)
            self.ar_list.SetStringItem(index=index, col=2, label=loss)
            index += 1
        self.ar_list.SetColumnWidth(col=0, width=wx.LIST_AUTOSIZE)
        self.ar_list.SetColumnWidth(col=1, width=wx.LIST_AUTOSIZE)
        self.ar_list.SetColumnWidth(col=2, width=wx.LIST_AUTOSIZE)

        # Set up a button and fields to add new, custom materials to the list
        self.new_mat_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.add_mat_btn = wx.Button(self, label="Add material")
        self.txt1 = PlaceholderTextCtrl(parent=self, placeholder="Material name")
        self.txt2 = PlaceholderTextCtrl(parent=self, placeholder="Dielectric")
        self.txt3 = PlaceholderTextCtrl(parent=self, placeholder="Loss tangent")
        self.new_mat_sizer.Add(self.txt1, flag=wx.RIGHT, border=3)
        self.new_mat_sizer.Add(self.txt2, flag=wx.RIGHT, border=3)
        self.new_mat_sizer.Add(self.txt3, flag=wx.RIGHT, border=3)
        self.new_mat_sizer.Add(self.add_mat_btn)

        # Arrange the list and material adding portions appropriately
        self.main_sizer = wx.GridBagSizer(hgap=5, vgap=5)
        self.main_sizer.Add(item=self.ar_list, pos=(0,0), flag=wx.EXPAND)
        self.main_sizer.Add(item=self.new_mat_sizer, pos=(1,0))

        self.SetSizer(self.main_sizer)
        self.main_sizer.Fit(self)

        # Set up event bindings for adding a new material
        self.Bind(wx.EVT_BUTTON, self.on_add_mat, self.add_mat_btn)
        self.Bind(wx.EVT_LIST_KEY_DOWN, self.on_delete_item, self.ar_list)

    def on_add_mat(self, event):
        material = self.txt1.GetValue()
        dielectric = self.txt2.GetValue()
        loss = self.txt3.GetValue()
        self.ar_list.Append((material, dielectric, loss))
        self.txt1.SetValue("Material name")
        self.txt2.SetValue("Dielectric")
        self.txt3.SetValue("Loss tangent")

    def on_delete_item(self, event):
        keycode = event.GetKeyCode()
        if keycode == wx.WXK_BACK:
            self.ar_list.DeleteItem(self.ar_list.GetFocusedItem())
        event.Skip()

#         # This is the beginning of list sorting functionality but it has a 
#         # long way to go.
#         listmix.ColumnSorterMixin.__init__(self, 3) # where 3 is number of columns
#         self.Bind(wx.EVT_LIST_COL_CLICK, self.on_col_click, self.ar_list)

    def GetListCtrl(self):
        """Required by ColumnSorterMixin, see see wx/lib/mixins/listctrl.py"""
        return self.ar_list

    def on_col_click(self, event):
        print "Column clicked."
        event.Skip()


class PlaceholderTextCtrl(wx.TextCtrl):
    def __init__(self, parent=None, placeholder=""):
        self.default_text = placeholder
        wx.TextCtrl.__init__(self, parent=parent, value=self.default_text)
        self.Bind(wx.EVT_SET_FOCUS, self.on_set_focus)
        self.Bind(wx.EVT_KILL_FOCUS, self.on_kill_focus)

    def on_set_focus(self, event):
        if self.GetValue() == self.default_text:
            self.SetValue("")
        event.Skip()

    def on_kill_focus(self, event):
        if self.GetValue().strip() == "":
            self.SetValue(self.default_text)
        event.Skip()


if __name__ == "__main__":
    app = wx.App(False)
    frame = MainWindow(None, "artools")
    app.MainLoop()
