# Importing functions separately, because it seems more organized.
from numpy import zeros, linspace, meshgrid, empty, rot90, array, round
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import (Entry, Label, StringVar, IntVar, Tk, Frame, Button, Toplevel,
                     filedialog, Checkbutton, Menu, messagebox, Scale,
                     SUNKEN, RAISED, HORIZONTAL)
from tkinter.font import Font
from tkinter.ttk import Combobox, Notebook, Style, Progressbar
from idlelib.tooltip import Hovertip

from numba import njit, prange, config  
#!!!At the time of sumbmiting this code, numba is not yet supported on python 3.12!!!
# Numba is a crucial for optimalization and this code won't run effectively without it, if at all.

from time import time
import sys
import subprocess
from threading import Thread
import os
from PIL import Image, ImageTk
import webbrowser

# config.DISABLE_JIT = True 
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# Uncomment to disable JIT Compilation, not reccomended.

# Calculating the pixel escape time / differance between z0 and last_z for color handling in the mandelbrot set
@njit(fastmath=True)
def mandelbrot_b(exponent, z0, c, Nb):
    last_z = z0 # Store the initial value of z0
    exponent = float(exponent)
    for i in prange(Nb): # If abs(c) > 2, no need for further calculation
        if abs(c) > 2:
            return i
        z = z0**exponent + c
        last_z = z0 # Update the last_z with the previous value of z0
        z0 = z
        if abs(z0) > 2:
            return i # Return the number of iterations it took to escape
        elif abs(z0 - last_z) == 0:
            return abs(z0 - last_z) + 1e-10
    return abs(z0 - last_z)

# Calculating the pixel color based on result from mandelbrot_b function in the mandelbrot set
@njit(parallel = True, fastmath=True)
def update_b_core(num_points, complex_grid, exponent, z0, Nb, colors, color_toggle1, color_toggle2, cmap1, cmap2, scale_factor,cmap1_default,cmap2_default):
    for i in prange(num_points):
        for j in prange(num_points):
            c = complex_grid[i, j]
            result = mandelbrot_b(exponent, z0, c, Nb)
            # Handle integer and fractional results for color assignment
            if result % 1 == 0:
                colors[i, j, :] = cmap1[int(result), :] if color_toggle1 else cmap1_default
            else:
                position = min(int(result * scale_factor), Nb - 1)
                colors[i, j, :] = cmap2[position, :] if color_toggle2 else cmap2_default
    return colors

# Calculating the pixel escape time / differance between z0 and last_z for color handling in the julia set
@njit(fastmath=True)
def julia_b( z0, c, Nb_j, exponent):
    last_z = z0  # Store the initial value of z0
    exponent = float(exponent)
    for i in prange(Nb_j):
        if abs(c) > 2:
            return i
        z = z0**exponent + c
        last_z = z0  # Update the last_z with the previous value of z0
        z0 = z
        if abs(z0) > 2:
            return i  # Return the number of iterations it took to escape
        elif abs(z0 - last_z) ==0:
            return abs(z0 - last_z)+10e-10
    return abs(z0 - last_z)

# Calculating the pixel color based on result from julia_b function in the mandelbrot set
@njit(parallel = True, fastmath=True)
def update_julia_core(c,num_points_j, complex_grid, exponent, z0, Nb_j, colors, color_toggle1, color_toggle2, cmap1, cmap2, scale_factor,cmap1_default,cmap2_default):
        for i in prange(num_points_j):
            for j in prange(num_points_j):
                z0 = complex_grid[i, j]
                result = julia_b(z0, c, Nb_j, exponent)
                # Handle integer and fractional results for color assignment
                if result % 1 == 0:
                    colors[i, j, :] = cmap1[int(result), :] if color_toggle1 else cmap1_default
                else:
                    position = min(int(result * scale_factor), Nb_j - 1)
                    colors[i, j,:] = cmap2[position, :] if color_toggle2 else cmap2_default
        return colors

# Calculating point coordinates for scatter plot
@njit(fastmath=True)
def mandelbrot(exponent, z0, c, N, xlim, ylim):
    if max(xlim[0], xlim[1], ylim[0], ylim[1]) > 2:
        limit = 2*max(xlim[0], xlim[1], ylim[0], ylim[1])
    else:
        limit = 2
    z = [0]
    for _ in prange(N):
        z0 = z0 ** exponent + c
        z.append(z0)
        if abs(z0) > limit: # Stop calculating if point escapes shown axis limits
            break
    return z

class MandelbrotPlotter:
    def __init__(self, window, entry_frame1):
        self.window = window
        self.t = time()
        self.exponent_def = 2
        self.exponent = self.exponent_def
        self.N_def = 100
        self.N = self.N_def
        self.Nb_def = 500
        self.Nb = self.Nb_def
        self.num_points_def = 500
        self.num_points = self.num_points_def
        self.Nb_j_def = 500
        self.Nb_j = self.Nb_j_def
        self.num_points_j_def = 500
        self.num_points_j = self.num_points_j_def
        self.z0_def = 0
        self.z0 = self.z0_def
        self.z = self.z0_def
        self.real = -0.5
        self.imag = 0.5
        self.c = complex(self.real, self.imag)
        self.xmin_def = -1.5
        self.xmin = self.xmin_def
        self.xmax_def = 0.5
        self.xmax = self.xmax_def
        self.ymin_def = -1.0
        self.ymin = self.ymin_def
        self.ymax_def = 1.0
        self.ymax = self.ymax_def
        self.xlim_j_def = -1.5, 1.5
        self.xlim_j = self.xlim_j_def
        self.ylim_j_def = -1.5, 1.5
        self.ylim_j = self.ylim_j_def
        self.factor = 0.5
        self.zoom_factor_def = 10
        self.zoom_factor = self.zoom_factor_def
        self.colors = zeros((self.num_points, self.num_points, 4))
        self.font_size = 14
        self.big_font = int(1.5*self.font_size)
        self.basic_font = Font(family="Arial", size=self.font_size)
        self.link_font = Font(family="Arial", size=self.font_size, underline=1)
        self.points_toggle = True
        self.lines_toggle = True
        self.entry_toggle = True
        self.julia_toggle = True
        self.grid_toggle = True
        self.thread_toggle = True
        self.shortcuts_toggle = True
        self.start_on_click = True
        self.start_colors = True
        self.start_point_color = True
        self.start_point_position = True
        self.start_julia_entry = True
        self.start_mandelbrot_entry = True
        self.start_jump = True
        self.start_switch = True
        self.start_zoom = True
        self.start_zoom_j = True
        self.run_task_button = True
        self.start_exponent_entry = True
        self.right_menu = None
        self.cmap1_temp = None
        self.cmap2_temp = None
        self.points_color = "red"
        self.line_color = "blue"
        self.cmap1_init = "plasma"
        self.cmap2_init = "viridis"
        self.line_width_def = 1
        self.line_width = self.line_width_def
        self.point_size_def = 5
        self.point_size = self.point_size_def
        validate_int = window.register(self.on_validate_int)
        validate_float = window.register(self.on_validate_float)

        self.window.bind('<Control-e>', self.run_task)
        self.window.bind('<Control-d>', self.default_entries)
        self.window.bind('<Control-z>', self.run_zoom_entry)
        self.window.bind('<Control-w>', self.switch_plots)
        self.window.bind('<Control-n>', self.new_window)
        self.window.bind('<Control-r>', self.restart)
        self.window.bind('<Control-m>', self.save_m)
        self.window.bind('<Control-j>', self.save_j)
        self.window.bind('<Control-h>', self.hint)
        self.window.bind('<Control-g>', self.about)
        self.window.bind('<F1>', self.shortcuts)
        self.window.bind('<Alt-q>', self.control_shortcuts)
        self.window.bind('<Escape>', self.focus)
        self.window.bind('<Button-3>', self.right_click_menu)
        self.window.bind('<Button-1>', self.close_right)
        # Create a Matplotlib figure and integrate it into the Tkinter window
        self.fig_m, self.ax = plt.subplots(figsize=(5, 5))  # Set the initial figure size 
        self.canvas_m = FigureCanvasTkAgg(self.fig_m, master=window)
        self.canvas_m.get_tk_widget().grid(row=0, column=0, padx=10, pady=10,sticky="nsew") 

        self.canvas_m.mpl_connect('scroll_event', self.zoom_around_cursor_m)
        self.canvas_m.mpl_connect('motion_notify_event', self.on_zoom_m)

        self.fig_j, self.ax_j = plt.subplots(figsize=(5, 5))
        plt.subplots_adjust(right = 1, bottom=0.1)
        self.canvas_j = FigureCanvasTkAgg(self.fig_j, master=entry_frame1)
        self.canvas_j.get_tk_widget().grid(row=5, column=0, padx=10, pady=10,sticky="w")
        plt.connect('scroll_event', self.zoom_around_cursor_j)
        self.canvas_j.mpl_connect('motion_notify_event', self.on_zoom_j)

        self.arr1 = linspace(0, self.Nb, self.Nb).reshape((self.Nb, -1))
        self.fig, self.ax1 = plt.subplots(figsize=(2, 1), ncols=1) 
        plt.subplots_adjust(top = 0.7, bottom = 0.5)
        rotated_arr = rot90(self.arr1)  # Rotate the array 90 degrees
        img = self.ax1.imshow(rotated_arr, interpolation='nearest', cmap=self.cmap1_init, aspect='auto') 
        self.ax1.set_yticks([])  # Disable x-axis
        self.ax1.set_xlabel("Iterations it took to escape",fontsize=int(0.7*self.font_size))
        self.ax1.set_title("Mandelbrot Set",fontsize=self.font_size)
        self.canvas1 = FigureCanvasTkAgg(self.fig, master=entry_frame1)
        self.canvas1.get_tk_widget().place(x=500, y=447)

        self.fig, self.ax2 = plt.subplots(figsize=(2, 1), ncols=1)
        plt.subplots_adjust(top = 0.7, bottom = 0.5)
        rotated_arr = rot90(self.arr1)  # Rotate the array 90 degrees
        img = self.ax2.imshow(rotated_arr, interpolation='nearest', cmap=self.cmap2_init, aspect='auto')
        self.ax2.set_yticks([])  # Disable x-axis
        self.ax2.set_xlabel("Difference of last iterations",fontsize=int(0.7*self.font_size))
        self.ax2.set_xticks([0, self.Nb//2, self.Nb])
        self.ax2.set_xticklabels([0, 1.5, 3])  # Set x-axis tick labels at 0, Nb/2, and Nb
        self.canvas2 = FigureCanvasTkAgg(self.fig, master=entry_frame1)
        self.canvas2.get_tk_widget().place(x=500, y=547)

        self.arr2 = linspace(0, self.Nb_j, self.Nb_j).reshape((self.Nb_j, -1))
        self.fig, self.ax3 = plt.subplots(figsize=(2, 1), ncols=1)  
        plt.subplots_adjust(top = 0.7, bottom = 0.5)
        rotated_arr = rot90(self.arr2)  # Rotate the array 90 degrees
        img = self.ax3.imshow(rotated_arr, interpolation='nearest', cmap=self.cmap1_init, aspect='auto') 
        self.ax3.set_yticks([])  # Disable x-axis
        self.ax3.set_xlabel("Iterations it took to escape",fontsize=int(0.7*self.font_size))
        self.ax3.set_title("Julia Set",fontsize=self.font_size)
        self.canvas3 = FigureCanvasTkAgg(self.fig, master=entry_frame1)
        self.canvas3.get_tk_widget().place(x=500, y=647)

        self.fig, self.ax4 = plt.subplots(figsize=(2, 1), ncols=1) 
        plt.subplots_adjust(top = 0.7, bottom = 0.5)
        rotated_arr = rot90(self.arr2)  # Rotate the array 90 degrees
        img = self.ax4.imshow(rotated_arr, interpolation='nearest', cmap=self.cmap2_init, aspect='auto')  
        self.ax4.set_yticks([])  # Disable x-axis
        self.ax4.set_xlabel("Difference of last iterations",fontsize=int(0.7*self.font_size))
        self.ax4.set_xticks([0, self.Nb_j//2, self.Nb_j])
        self.ax4.set_xticklabels([0, 1.5, 3])  
        self.canvas4 = FigureCanvasTkAgg(self.fig, master=entry_frame1)
        self.canvas4.get_tk_widget().place(x=500, y=747)

        self.toolbar1 = NavigationToolbar2Tk(self.canvas_m, window,pack_toolbar=False)
        self.toolbar1.update()
        self.toolbar1.place(x = 0, y = 50)

        self.toolbar2 = NavigationToolbar2Tk(self.canvas_j, entry_frame1,pack_toolbar=False)
        self.toolbar2.update()
        self.toolbar2.place(x = 0, y = 447)

        self.create_cmap_dropdowns()
        self.z = mandelbrot(self.exponent, self.z0, self.c, self.N, self.ax.get_xlim(), self.ax.get_ylim())
        self.real_parts = [z_i.real for z_i in self.z]
        self.imaginary_parts = [z_i.imag for z_i in self.z]
        
        self.line, = self.ax.plot(self.real_parts, self.imaginary_parts, linewidth=self.line_width,
                                    color = mcolors.to_rgba(self.line_color))
        self.scatter = self.ax.scatter(self.real_parts, self.imaginary_parts, s=self.point_size,
                                    color = mcolors.to_rgba(self.points_color))
        self.scatter_c = self.ax.scatter(self.real_parts[1], self.imaginary_parts[1], s=10*self.point_size,
                                    color = mcolors.to_rgba(self.points_color))
        
        self.ax.set_title("Mandelbrot Set",fontsize=int(2*self.font_size))
        self.title_j = self.ax_j.set_title("Julia Set",fontsize=self.big_font)
        self.title_j.set_position([0.85, 1])
        self.ax.set_xlabel("Real Axis",fontsize=self.big_font)
        self.ax.set_ylabel("Imaginary Axis",fontsize=self.big_font)
        self.ax_j.set_xlabel('Real Axis',fontsize=int(self.font_size))
        self.ax_j.set_ylabel('Imaginary Axis',fontsize=int(self.font_size))
        self.ax.set_xlim(self.xmin, self.xmax)
        self.ax.set_ylim(self.ymin, self.ymax)

        # Create input fields and labels
        self.progress_var = IntVar()
        self.progress_var.set(100)
        self.progress_bar = Progressbar(window, variable=self.progress_var, maximum=100)
        self.progress_bar.place(x = 20, y = 100)
        N_label = Label(entry_frame1, text="Iterations for c:", 
                                font=self.basic_font)
        N_Tip = Hovertip(N_label,f'Default is {self.N_def}', hover_delay=None)

        real_label = Label(entry_frame1, text="Real part of c:", 
                                font=self.basic_font)
        real_Tip = Hovertip(real_label,'Default is -0.5', hover_delay=None)

        imag_label = Label(entry_frame1, text="Imaginary part of c:", 
                                font=self.basic_font)
        imag_Tip = Hovertip(imag_label,'Default is 0.5', hover_delay=None)

        num_points_label = Label(entry_frame1, text="Mandelbrot grid size:", 
                                font=self.basic_font)
        num_points_Tip = Hovertip(num_points_label,f'Default is {self.num_points_def}', hover_delay=None)

        Nb_label = Label(entry_frame1, text="Mandelbrot iterations:", font=self.basic_font)
        Nb_Tip = Hovertip(Nb_label,f'Default is {self.Nb_def}', hover_delay=None)

        xmin_label = Label(entry_frame1, text="Minimum real axis :", 
                                font=self.basic_font)
        xmin_Tip = Hovertip(xmin_label,f'Default is {self.xmin_def}', hover_delay=None)

        xmax_label = Label(entry_frame1, text="Maximum real axis :", 
                                font=self.basic_font)
        xmax_Tip = Hovertip(xmax_label,f'Default is {self.xmax_def}', hover_delay=None)

        ymin_label = Label(entry_frame1, text="Minimum imaginary axis :", 
                                font=self.basic_font)
        ymin_Tip = Hovertip(ymin_label,f'Default is {self.ymin_def}', hover_delay=None)

        ymax_label = Label(entry_frame1, text="Maximum imaginary axis :", 
                                font=self.basic_font)
        ymax_Tip = Hovertip(ymax_label,f'Default is {self.ymax_def}', hover_delay=None)

        exponent_label = Label(entry_frame1, text="Exponent :", 
                                font=self.basic_font)
        exponent_Tip = Hovertip(exponent_label,f'Default is {self.exponent_def}', hover_delay=None)

        point_size_label = Label(entry_frame1, text="Point size:", 
                                font=self.basic_font)
        point_size_Tip = Hovertip(point_size_label,f'Default is {self.point_size_def}', hover_delay=None)

        linewidth_label = Label(entry_frame1, text="Line width:", 
                                font=self.basic_font)
        linewidth_Tip = Hovertip(linewidth_label,f'Default is {self.line_width_def}', hover_delay=None)

        zoom_label = Label(entry_frame1, text="Zoom by:", 
                                font=self.basic_font)
        zoom_Tip = Hovertip(zoom_label,f'Default is {self.zoom_factor_def}', hover_delay=None)

        self.result_label = Label(entry_frame1, text="", width = 40, font=("Arial", int(1.2*self.font_size)))

        self.loaded_label = Label(window, text="Loaded", font=("Arial", self.big_font), fg="lime",bg="white",)
        self.function_label = Label(entry_frame1, text="", pady = 10, font=("Arial", int(1.2*self.font_size)))
        function_label_Tip = Hovertip(self.function_label,'z0 = 0', hover_delay=None)

        self.shortcut_label = Label(entry_frame1, text="", pady = 10, font=("Arial", int(0.9*self.font_size)), fg = "red")

        self.N_entry = Entry(entry_frame1, font=self.basic_font,width=7,validate="key", validatecommand=(validate_int, "%P"))
        
        self.real_entry = Entry(entry_frame1, font=self.basic_font,validate="key", validatecommand=(validate_float, "%P"))
        self.imag_entry = Entry(entry_frame1, font=self.basic_font,validate="key", validatecommand=(validate_float, "%P"))
        self.num_points_entry = Entry(entry_frame1, font=self.basic_font, width=7,validate="key", validatecommand=(validate_int, "%P"))
        self.Nb_entry = Entry(entry_frame1, font=self.basic_font, width=7,validate="key", validatecommand=(validate_int, "%P"))

        self.xmin_entry = Entry(entry_frame1, font=self.basic_font,validate="key", validatecommand=(validate_float, "%P"))
        self.xmax_entry = Entry(entry_frame1, font=self.basic_font,validate="key", validatecommand=(validate_float, "%P"))
        self.ymin_entry = Entry(entry_frame1, font=self.basic_font,validate="key", validatecommand=(validate_float, "%P"))
        self.ymax_entry = Entry(entry_frame1, font=self.basic_font,validate="key", validatecommand=(validate_float, "%P"))

        self.exponent_entry = Entry(entry_frame1, font=self.basic_font,width=7,validate="key", validatecommand=(validate_float, "%P"))
        self.point_size_entry = Entry(entry_frame1, font=self.basic_font,width=5,validate="key", validatecommand=(validate_float, "%P"))
        self.line_width_entry = Entry(entry_frame1, font=self.basic_font,width=5,validate="key", validatecommand=(validate_float, "%P"))
        self.zoom_entry = Entry(entry_frame1, font=self.basic_font,width=5,validate="key", validatecommand=(validate_float, "%P"))

        Nb_j_label = Label(entry_frame1, text="Julia iterations:",font=self.basic_font)
        Nb_j_Tip = Hovertip(Nb_j_label,f'Default is {self.Nb_j_def}', hover_delay=None)
        self.Nb_j_entry = Entry(entry_frame1, font=self.basic_font,width=7,validate="key", validatecommand=(validate_int, "%P"))

        num_points_j_label = Label(entry_frame1, text="Julia grid size:",font=self.basic_font)
        num_points_j_Tip = Hovertip(num_points_j_label,f'Default is {self.num_points_j_def}', hover_delay=None)
        self.num_points_j_entry = Entry(entry_frame1, font=self.basic_font,width=7,validate="key", validatecommand=(validate_int, "%P"))

        run_button = Button(entry_frame1, text="Run", 
                        command=self.run_task,
                        font=("Arial", self.big_font),
                        bg = "lime", activebackground="lime"
                        )
        run_button_Tip = Hovertip(run_button,'Updates all parameters', hover_delay=1000)

        restart_button = Button(entry_frame1, text="Restart", 
                                command=self.restart,
                                font=self.basic_font,
                                bg = "red", fg = "white",
                                activebackground = "red", activeforeground = "white"
                                )
        restart_Tip = Hovertip(restart_button,'Restarts the app', hover_delay=0)

        self.freeze_julia_button = Checkbutton(entry_frame1, text="Freeze julia", 
                        command=self.freeze_julia,
                        font=self.basic_font,
                        relief = RAISED,
                        bg ="yellow", fg ="black",
                        activebackground = "yellow", activeforeground = "black"
                        )
        freeze_julia_Tip = Hovertip(self.freeze_julia_button,'Stops the colormap of Julia set\nfrom updating', hover_delay=500)


        self.toggle_points_button = Checkbutton(entry_frame1, text="Hide points", 
                                command=self.update_points,
                                font=self.basic_font,
                                relief = RAISED,
                                bg ="yellow", fg ="black",
                                activebackground = "yellow", activeforeground = "black",
                                )

        self.toggle_lines_button = Checkbutton(entry_frame1, text="Hide lines", 
                                command=self.update_lines,
                                font=self.basic_font,
                                relief= RAISED,
                                bg ="yellow", fg ="black",
                                activebackground = "yellow", activeforeground = "black",
                                )
        
        self.hide_grid_button = Checkbutton(entry_frame1, text="Grid", 
                                command=self.hide_grid,
                                font=self.basic_font,
                                relief= RAISED,
                                bg ="yellow", fg ="black",
                                activebackground = "yellow", activeforeground = "black",
                                )
        
        switch_button = Button(window, text="Switch plots", 
                                command=self.switch_plots,
                                font=self.basic_font,
                                bg ="blue", fg ="white",
                                activebackground = "blue", activeforeground = "white"
                                )
        switch_Tip = Hovertip(switch_button,'Makes the two plots\nswitch places', hover_delay=500)
        
        zoom_button = Button(entry_frame1, text="Zoom to point", 
                                command=self.zoom_to_point,
                                font=self.basic_font,
                                bg ="blue", fg ="white",
                                activebackground = "blue", activeforeground = "white"
                                )
        zoom_Tip = Hovertip(zoom_button,'Zooms to placed point c,\nzoom out if zoom factor is less than 1', hover_delay=500)
    
        default_button = Button(entry_frame1, text="Default values", 
                                command=self.default_entries,
                                font=self.basic_font,
                                bg = "blue",fg = "white",
                                activebackground = "blue", activeforeground = "white"
                                )
        self.zoom_scale = Scale(entry_frame1, resolution=0.01, from_=0, to=100, orient=HORIZONTAL, command=self.update_zoom_value, length=100)
        self.zoom_scale.grid(row=3, column=0, padx=100,in_=tab1)
        self.zoom_scale.bind("<MouseWheel>", lambda event, scale=self.zoom_scale: self.on_scroll(scale, event))

        self.N_scale = Scale(entry_frame1, from_=2, to=2000, orient=HORIZONTAL, command=self.update_N_value, length=300)
        self.N_scale.grid(row=0, column=1, padx=(100, 0), in_=tab2)
        self.N_scale.bind("<MouseWheel>", lambda event, scale=self.N_scale: self.on_scroll(scale, event))

        self.point_scale = Scale(entry_frame1, from_=0, to=100, orient=HORIZONTAL, command=self.update_point_value, length=300)
        self.point_scale.grid(row=3, column=1, padx=(100, 0), in_=tab2)
        self.point_scale.bind("<MouseWheel>", lambda event, scale=self.point_scale: self.on_scroll(scale, event))
        
        self.line_scale = Scale(entry_frame1, from_=0, to=100, orient=HORIZONTAL, command=self.update_line_value, length=300)
        self.line_scale.grid(row=4, column=1, padx=(100, 0), in_=tab2)
        self.line_scale.bind("<MouseWheel>", lambda event, scale=self.line_scale: self.on_scroll(scale, event))
        
        self.num_points_scale = Scale(entry_frame1, from_=2, to=2000, orient=HORIZONTAL, command=self.update_num_points_value, length=300)
        self.num_points_scale.grid(row=0, column=1, padx=(100, 45), in_=tab3)
        self.num_points_scale.bind("<MouseWheel>", lambda event, scale=self.num_points_scale: self.on_scroll(scale, event))
        
        self.Nb_scale = Scale(entry_frame1, from_=2, to=2000, orient=HORIZONTAL, command=self.update_Nb_value, length=300)
        self.Nb_scale.grid(row=1, column=1, padx=(100, 45), in_=tab3)
        self.Nb_scale.bind("<MouseWheel>", lambda event, scale=self.Nb_scale: self.on_scroll(scale, event))
        
        self.num_points_j_scale = Scale(entry_frame1, from_=2, to=2000, orient=HORIZONTAL, command=self.update_num_points_j_value, length=300)
        self.num_points_j_scale.grid(row=2, column=1, padx=(100, 45), in_=tab3)
        self.num_points_j_scale.bind("<MouseWheel>", lambda event, scale=self.num_points_j_scale: self.on_scroll(scale, event))
        
        self.Nb_j_scale = Scale(entry_frame1, from_=2, to=2000, orient=HORIZONTAL, command=self.update_Nb_j_value, length=300)
        self.Nb_j_scale.grid(row=3, column=1, padx=(100, 45), in_=tab3)
        self.Nb_j_scale.bind("<MouseWheel>", lambda event, scale=self.Nb_j_scale: self.on_scroll(scale, event))
        
        self.exponent_scale = Scale(entry_frame1, resolution=0.1, from_=1, to=10, orient=HORIZONTAL, command=self.update_exponent_value, length=300)
        self.exponent_scale.grid(row=4, column=1, padx=(100, 45), sticky = "e", in_=tab3)
        self.exponent_scale.bind("<MouseWheel>", lambda event, scale=self.exponent_scale: self.on_scroll(scale, event))
        
        self.style = Style()
        self.style.configure('TNotebook.Tab', font=('Arial', int(1.5*self.font_size))) 

        # Menu bar
        menu_bar = Menu(window)
        window.config(menu=menu_bar)

        # File menu
        file_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Window (Ctrl + N)", command=self.new_window)
        file_menu.add_command(label="Save Mandelbrot (Ctrl + M)", command=self.save_m)
        file_menu.add_command(label="Save Julia (Ctrl + J)", command=self.save_j)
        file_menu.add_command(label="Close (Alt + F4)", command=self.close)

        # Help menu
        help_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Hint (Ctrl + H)", command=self.hint)
        help_menu.add_command(label="About (Ctrl + A)", command=self.about)
        help_menu.add_command(label="List Of Shortcuts (Ctrl + T)", command=self.shortcuts)

        # Grid layout for widgets
        x_pos = 480
        y_pos = 20
        self.loaded_label.place(x=10, y=10, in_=window)
        self.function_label.grid(row=0, column=0, padx=80, pady=10, sticky="w", in_=tab1)
        self.result_label.grid(row=1, column=0, padx=0, sticky="w", pady=10, in_=tab1)
        run_button.grid(row=2, column=0, padx=80, pady=10, in_=tab1)
        self.shortcut_label.grid(row=2, column=0, padx=(100,0), pady=5, sticky="w", in_=tab1)
        zoom_label.grid(row=3, column=0, padx=10, pady=5, sticky="w", in_=tab1)
        self.zoom_entry.grid(row=3, column=0, padx=(180,0), pady=5, sticky="w", in_=tab1)

        self.toggle_points_button.place(x=x_pos, y=y_pos, in_=tab1)
        self.toggle_lines_button.place(x=x_pos, y=y_pos + 50, in_=tab1)
        self.hide_grid_button.place(x=x_pos, y=y_pos + 100, in_=tab1)
        self.freeze_julia_button.place(x=x_pos, y=y_pos + 150, in_=tab1)
        default_button.place(x=x_pos, y=y_pos + 200, in_=tab1)
        zoom_button.place(x=x_pos, y=y_pos + 250, in_=tab1)
        switch_button.place(x=x_pos, y=y_pos + 300, in_=tab1)

        restart_button.place(x=x_pos + 50, y=y_pos + 850, in_=entry_frame1)

        N_label.grid(row=0, column=0, padx=10, pady=5, sticky="e", in_=tab2)
        self.N_entry.grid(row=0, column=1, padx=10, pady=5, in_=tab2, sticky="w")
        real_label.grid(row=1, column=0, padx=10, pady=5, sticky="e", in_=tab2)
        self.real_entry.grid(row=1, column=1, padx=10, pady=5, in_=tab2, sticky="w")
        imag_label.grid(row=2, column=0, padx=10, pady=5, sticky="e", in_=tab2)
        self.imag_entry.grid(row=2, column=1, padx=10, pady=5, in_=tab2, sticky="w")
        point_size_label.grid(row=3, column=0, padx=10, pady=5, sticky="e", in_=tab2)
        self.point_size_entry.grid(row=3, column=1, padx=10, pady=5, in_=tab2, sticky="w")
        linewidth_label.grid(row=4, column=0, padx=10, pady=5, sticky="e", in_=tab2)
        self.line_width_entry.grid(row=4, column=1, padx=10, pady=5, in_=tab2, sticky="w")

        num_points_label.grid(row=0, column=0, padx=10, pady=5, sticky="e", in_=tab3)
        self.num_points_entry.grid(row=0, column=1, padx=10, pady=5, in_=tab3, sticky="w")
        Nb_label.grid(row=1, column=0, padx=10, pady=5, sticky="e", in_=tab3)
        self.Nb_entry.grid(row=1, column=1, padx=10, pady=5, in_=tab3, sticky="w")
        num_points_j_label.grid(row=2, column=0, padx=10, pady=5, sticky = "e", in_=tab3)
        self.num_points_j_entry.grid(row=2, column=1, padx=10, pady=5, in_=tab3, sticky = "w")
        Nb_j_label.grid(row=3, column=0, padx=10, pady=5, sticky = "e", in_=tab3)
        self.Nb_j_entry.grid(row=3, column=1, padx=10, pady=5, in_=tab3, sticky = "w")
        exponent_label.grid(row=4, column=0, padx=10, pady=5, sticky="e", in_=tab3)
        self.exponent_entry.grid(row=4, column=1, padx=10, pady=5, in_=tab3, sticky="w")
        xmin_label.grid(row=5, column=0, padx=10, pady=5, sticky="e", in_=tab3)
        self.xmin_entry.grid(row=5, column=1, padx=10, pady=5, in_=tab3, sticky="w")
        xmax_label.grid(row=6, column=0, padx=10, pady=5, sticky="e", in_=tab3)
        self.xmax_entry.grid(row=6, column=1, padx=10, pady=5, in_=tab3, sticky="w")
        ymin_label.grid(row=7, column=0, padx=10, pady=5, sticky="e", in_=tab3)
        self.ymin_entry.grid(row=7, column=1, padx=10, pady=5, in_=tab3, sticky="w")
        ymax_label.grid(row=8, column=0, padx=10, pady=5, sticky="e", in_=tab3)
        self.ymax_entry.grid(row=8, column=1, padx=10, pady=5, in_=tab3, sticky="w")

        # Binding entries to callbacks
        self.N_entry.bind("<Return>",self.run_point_position)
        self.real_entry.bind("<Return>",self.run_point_position)
        self.imag_entry.bind("<Return>",self.run_point_position)
        self.point_size_entry.bind("<Return>",self.run_point_color)
        self.line_width_entry.bind("<Return>",self.run_point_color)
        self.zoom_entry.bind("<Return>",self.run_zoom_entry)
        
        self.num_points_entry.bind("<Return>",self.run_mandelbrot_entry)
        self.Nb_entry.bind("<Return>",self.run_mandelbrot_entry)
        self.xmin_entry.bind("<Return>",self.run_mandelbrot_entry)
        self.xmax_entry.bind("<Return>",self.run_mandelbrot_entry)
        self.ymin_entry.bind("<Return>",self.run_mandelbrot_entry)
        self.ymax_entry.bind("<Return>",self.run_mandelbrot_entry)
        self.exponent_entry.bind("<Return>",self.run_exponent_entry)
        
        self.num_points_j_entry.bind("<Return>",self.run_julia_entry)
        self.Nb_j_entry.bind("<Return>",self.run_julia_entry)

        self.canvas_m.mpl_connect("button_press_event", self.on_click)
        self.default_entries()
        self.run_mandelbrot()
        # self.hint()
        
    def right_click_menu(self, event):
        if self.right_menu and self.right_menu.winfo_exists():
            # If the menu already exists and is not destroyed, move it to the cursor position
            x, y = self.window.winfo_pointerx(), self.window.winfo_pointery()
            self.right_menu.geometry(f"+{x}+{y}")
        else:
            # If the menu doesn't exist or is destroyed, create a new one
            x, y = self.window.winfo_pointerx(), self.window.winfo_pointery()
            self.right_menu = Toplevel(self.window)
            self.right_menu.title("")
            self.right_menu.geometry(f"+{x}+{y}")
            self.right_menu.bind('<Button-1>', self.close_right)

            # List of labels and their corresponding functions
            label_data = [
                ("Ctrl + E - Updates all parameters", self.run_task),
                ("Ctrl + D - Sets the values to default", self.default_entries),
                ("Ctrl + Z - Zooms to placed point", self.run_zoom_entry),
                ("Ctrl + W - Switches plots", self.switch_plots),
                ("Ctrl + N - Opens new window", self.new_window),
                ("Ctrl + R - Restarts the app", self.restart),
                ("Ctrl + M - Brings the Mandelbrot set save dialog", self.save_m),
                ("Ctrl + J - Brings the Julia set save dialog", self.save_j),
                ('Ctrl + H - Shows the "Hint" message', self.hint),
                ('Ctrl + G - Shows the "About" message', self.about),
                ("Alt + F4 - Closes the app", self.close),
                ("Alt + Q - Disables Ctrl in other shortcuts", self.control_shortcuts)
            ]

            # Create labels using a loop
            for label_text, function in label_data:
                label = Label(self.right_menu, text=label_text, font = self.basic_font, cursor="hand2")
                label.pack(anchor="w")
                label.bind("<Button-1>", lambda event, f=function: f())

                label.bind("<Enter>", self.label_enter)
                label.bind("<Leave>", self.label_leave)

    def label_enter(self, event):
        # Change label background color to blue when the cursor enters
        event.widget.config(bg="light gray")

    def label_leave(self, event):
        # Change label background color back to default when the cursor leaves
        event.widget.config(bg="SystemButtonFace")
    
    def close_right(self,event):
        try:
            self.right_menu.destroy()
        except:
            pass

    def on_validate_float(self, P):
        # P is the proposed input
        if P == "":
            return True
        try:
            float(P)
            return True
        except ValueError:
            try:
                # Check if P is a minus sign
                if P == "-" and len(P) == 1:
                    return True
                # Check if P starts with a minus sign and the rest is a valid float
                elif P.startswith("-") and len(P) > 1:
                    float(P[1:])
                    return True
                else:
                    return False
            except ValueError:
                return False

    def on_validate_int(self,P):
        # P is the proposed input
        if P.isdigit() or P == "":
            return True
        else:
            return False

    # Using a thread to make loading progress possible
    def start_task(self):
        if self.thread_toggle:
            self.thread_toggle = False
            self.progress_var.set(0)  # Reset the progress bar
            task_thread = Thread(target=self.run_mandelbrot, args=(self.progress_var,))
            task_thread.start()

    def on_scroll(self,scale_widget, event):

        current_value = scale_widget.get()
        factor = 1 if scale_widget["to"]>500 else 10
        if scale_widget.winfo_name() == "!scale9":
            factor = 1000
        delta = event.delta
        new_value = current_value - delta/(1.2*factor)
        scale_widget.set(new_value)  

    def run_task(self,*args):
        self.run_task_button = True
        self.start_task()

    def run_point_position(self,event):
        self.start_point_position = True
        self.start_task()

    def run_point_color(self,event):
        self.start_point_color = True
        self.start_task()

    def run_julia_entry(self,event):
        self.start_julia_entry = True
        self.start_task()

    def run_mandelbrot_entry(self,event):
        self.start_mandelbrot_entry = True
        self.start_task()

    def run_exponent_entry(self,event):
        self.start_exponent_entry = True
        self.start_task()

    def run_zoom_entry(self,*args):
        self.zoom_to_point()

    def focus(self,*args):
        self.window.focus_set()

    def update_N_value(self,value):
        self.N_entry.delete(0,"end")
        self.N_entry.insert(0,value)
        self.N_entry.focus_set()
    
    def update_Nb_value(self,value):
        self.Nb_entry.delete(0,"end")
        self.Nb_entry.insert(0,value)
        self.Nb_entry.focus_set()
    
    def update_Nb_j_value(self,value):
        self.Nb_j_entry.delete(0,"end")
        self.Nb_j_entry.insert(0,value)
        self.Nb_j_entry.focus_set()
    
    def update_num_points_value(self,value):
        self.num_points_entry.delete(0,"end")
        self.num_points_entry.insert(0,value)
        self.num_points_entry.focus_set()
    
    def update_num_points_j_value(self,value):
        self.num_points_j_entry.delete(0,"end")
        self.num_points_j_entry.insert(0,value)
        self.num_points_j_entry.focus_set()
    
    def update_exponent_value(self,value):
        self.exponent_entry.delete(0,"end")
        self.exponent_entry.insert(0,value)
        self.exponent_entry.focus_set()

    def update_zoom_value(self,value):
        self.zoom_entry.delete(0,"end")
        self.zoom_entry.insert(0,value)
        self.zoom_entry.focus_set()

    def update_point_value(self,value):
        self.point_size_entry.delete(0,"end")
        self.point_size_entry.insert(0,value)
        self.point_size_entry.focus_set()

    def update_line_value(self,value):
        self.line_width_entry.delete(0,"end")
        self.line_width_entry.insert(0,value)
        self.line_width_entry.focus_set()

    def close(self,*args):
        self.window.destroy()
        
    def restart(self,*args):
        self.window.destroy()

        # Get the script name and create the full path
        script_name = sys.argv[0]
        script_path = os.path.abspath(script_name)

        # Start a new process with the script
        subprocess.Popen([sys.executable, script_path])

    def new_window(self,*args):

        # Get the script name and create the full path
        script_name = sys.argv[0]
        script_path = os.path.abspath(script_name)

        # Start a new process with the script
        subprocess.Popen([sys.executable, script_path])
        
    def hide_grid(self):
        self.t = time()
        self.grid_toggle = not self.grid_toggle
        if not self.grid_toggle:
            self.ax.grid()
            self.fig = self.ax.get_figure()
            self.canvas_m.draw()
            self.hide_grid_button.config(relief = SUNKEN)
        else:
            self.ax.grid(False)
            self.fig = self.ax.get_figure()
            self.canvas_m.draw()
            self.hide_grid_button.config(relief = RAISED)

    def freeze_julia(self):
        self.t = time()
        self.julia_toggle = not self.julia_toggle
        self.cmap1_temp = self.cmap1_var.get()
        self.cmap2_temp = self.cmap2_var.get()
        if self.julia_toggle:
            self.freeze_julia_button.config(relief = RAISED)
            self.update_julia()
        else:
            self.freeze_julia_button.config(relief = SUNKEN)

    def default_entries(self,*args):
        self.clear_entries()
        self.jump_var.set("Home")
        self.cmap1_var.set("plasma")
        self.cmap2_var.set("viridis")
        self.point_color_var.set("red")
        self.line_color_var.set("blue")
        self.N_entry.insert(0,str(self.N_def))

        self.point_size_entry.insert(0,str(5))
        self.line_width_entry.insert(0,str(1))

        self.exponent_entry.insert(0,str(self.exponent_def))

        self.num_points_entry.insert(0,str(self.num_points_def))
        self.Nb_entry.insert(0,str(self.Nb_def))

        self.num_points_j_entry.insert(0,str(self.num_points_j_def))
        self.Nb_j_entry.insert(0,str(self.Nb_j_def))
        self.zoom_entry.insert(0,str(10))

        self.real_entry.insert(0,str(-0.5))
        self.imag_entry.insert(0,str(0.5))
        self.xmin_entry.insert(0,str(self.xmin_def))
        self.xmax_entry.insert(0,str(self.xmax_def))
        self.ymin_entry.insert(0,str(self.ymin_def))
        self.ymax_entry.insert(0,str(self.ymax_def))

        self.xlim_j = self.xlim_j_def
        self.ylim_j = self.ylim_j_def

        self.N_scale.set(self.N_def)
        self.N_scale.configure(to=2000)

        self.num_points_scale.set(self.num_points_def)
        self.num_points_scale.configure(to=2000)

        self.Nb_scale.set(self.Nb_def)
        self.Nb_scale.configure(to=2000)

        self.num_points_j_scale.set(self.num_points_j_def)
        self.num_points_j_scale.configure(to=2000)

        self.Nb_j_scale.set(self.Nb_j_def)
        self.Nb_j_scale.configure(to=2000)

        self.exponent_scale.set(self.exponent_def)
        self.exponent_scale.configure(to=100)

        self.zoom_scale.set(10)
        self.zoom_scale.configure(to=100)

        self.point_scale.set(5)
        self.point_scale.configure(to=100)

        self.line_scale.set(1)
        self.line_scale.configure(to=100)

    def zoom_around_cursor_m(self, event):
        if self.thread_toggle == True:
            self.factor = 0.5 if event.step > 0 else 2  # Use event.step to check for scroll direction
            # Get the current axis limits
            self.xlim, self.ylim = self.ax.get_xlim(), self.ax.get_ylim()

            # Calculate the cursor position
            self.x_center = event.xdata
            self.y_center = event.ydata
            self.zoom()

    def zoom_around_cursor_j(self, event):
        if self.thread_toggle == True:
            try:
                self.t = time()
                self.factor = 0.5 if event.step > 0 else 2  # Use event.step to check for scroll direction
                # Get the current axis limits
                self.xlim, self.ylim = self.ax_j.get_xlim(), self.ax_j.get_ylim()

                # Calculate the cursor position 
                self.x_center = event.xdata
                self.y_center = event.ydata

                self.xlim_j = (self.x_center + (self.xlim[0] - self.x_center) * self.factor, 
                            self.x_center + (self.xlim[1] - self.x_center) * self.factor)
                self.ylim_j = (self.y_center + (self.ylim[0] - self.y_center) * self.factor, 
                            self.y_center + (self.ylim[1] - self.y_center) * self.factor)
                self.start_zoom_j = True
                self.start_task()
            except:
                pass

    def zoom_to_point(self):
        zoom_value = self.zoom_entry.get()
        try:
            self.zoom_factor = float(zoom_value) if float(zoom_value) > 0 else 10
        except:
            pass

        try:
            self.factor = 1/self.zoom_factor
        except:
            self.factor = 1

        self.xlim, self.ylim = self.ax.get_xlim(), self.ax.get_ylim()

        self.real_part_value = self.real_entry.get()
        self.real_part_value = float(self.real_part_value) if self.real_part_value else 0

        self.imag_part_value = self.imag_entry.get()
        self.imag_part_value = float(self.imag_part_value) if self.imag_part_value else 0

        self.x_center = self.real_part_value
        self.y_center = self.imag_part_value

        self.zoom()

    def zoom(self):
        # Calculate the new axis limits based on the zoom factor
        try:
            new_xlim = (self.x_center + (self.xlim[0] - self.x_center) * self.factor, 
                        self.x_center + (self.xlim[1] - self.x_center) * self.factor)
            new_ylim = (self.y_center + (self.ylim[0] - self.y_center) * self.factor, 
                        self.y_center + (self.ylim[1] - self.y_center) * self.factor)

            self.xmin_entry.delete(0, "end")
            self.xmax_entry.delete(0, "end")
            self.ymin_entry.delete(0, "end")
            self.ymax_entry.delete(0, "end")

            self.xmin_entry.insert(0,str(new_xlim[0]))
            self.xmax_entry.insert(0,str(new_xlim[1]))
            self.ymin_entry.insert(0,str(new_ylim[0]))
            self.ymax_entry.insert(0,str(new_ylim[1]))
            self.start_zoom = True
            self.start_task()
        except:
            pass

    def switch_plots(self,*args):
        self.canvas_m.get_tk_widget().destroy()
        self.canvas_j.get_tk_widget().destroy()
        self.entry_toggle = not self.entry_toggle
        if self.entry_toggle:
            self.fig_m, self.ax = plt.subplots(figsize=(5, 5))
            self.canvas_m = FigureCanvasTkAgg(self.fig_m, master=window)
            self.canvas_m.get_tk_widget().grid(row=0, column=0, padx=10, pady=10,sticky=("nsew"))
            self.canvas_m.mpl_connect('scroll_event', self.zoom_around_cursor_m)
            self.fig_m = self.ax.get_figure() 
            self.canvas_m.mpl_connect('motion_notify_event', self.on_zoom_m)
            self.canvas_m.mpl_connect('button_press_event', self.on_click)

            self.fig_j, self.ax_j = plt.subplots(figsize=(5, 5))
            self.canvas_j = FigureCanvasTkAgg(self.fig_j, master=entry_frame1)
            self.canvas_j.get_tk_widget().grid(row=3, column=0, padx=10, pady=10,sticky="w")
            self.canvas_j.mpl_connect('scroll_event', self.zoom_around_cursor_j)
            self.canvas_j.mpl_connect('motion_notify_event', self.on_zoom_j)
            plt.subplots_adjust(right = 1, bottom=0.2)

            self.ax.set_title("Mandelbrot Set",fontsize=int(2*self.font_size))
            self.title_j = self.ax_j.set_title("Julia Set",fontsize=self.big_font)
            self.title_j.set_position([0.85, 1])
            self.ax.set_xlabel("Real Axis",fontsize=self.big_font)
            self.ax.set_ylabel("Imaginary Axis",fontsize=self.big_font)
            self.ax_j.set_xlabel("Real Axis",fontsize=int(self.font_size))
            self.ax_j.set_ylabel("Imaginary Axis",fontsize=int(self.font_size))

            self.toolbar1 = NavigationToolbar2Tk(self.canvas_m, window,pack_toolbar=False)
            self.toolbar2 = NavigationToolbar2Tk(self.canvas_j, entry_frame1,pack_toolbar=False)
        else:
            self.fig_m, self.ax = plt.subplots(figsize=(5, 5))
            self.canvas_m = FigureCanvasTkAgg(self.fig_m, master=entry_frame1)
            self.canvas_m.get_tk_widget().grid(row=3, column=0, padx=10, pady=10,sticky="w")
            self.canvas_m.mpl_connect('scroll_event', self.zoom_around_cursor_m)
            plt.subplots_adjust(right = 1, bottom=0.2)
            self.fig_m = self.ax.get_figure()
            self.canvas_m.mpl_connect('motion_notify_event', self.on_zoom_m)
            self.canvas_m.mpl_connect('button_press_event', self.on_click)

            self.fig_j, self.ax_j = plt.subplots(figsize=(5, 5))
            self.canvas_j = FigureCanvasTkAgg(self.fig_j, master=window)
            self.canvas_j.get_tk_widget().grid(row=0, column=0, padx=10, pady=10,sticky=("nsew"))
            self.canvas_j.mpl_connect('scroll_event', self.zoom_around_cursor_j)
            self.canvas_j.mpl_connect('motion_notify_event', self.on_zoom_j)

            self.title_m = self.ax.set_title("Mandelbrot Set",fontsize=self.big_font)
            self.title_m.set_position([0.75, 1])
            self.ax_j.set_title("Julia Set",fontsize=int(2*self.font_size))
            self.ax.set_xlabel("Real Axis",fontsize=int(self.font_size))
            self.ax.set_ylabel("Imaginary Axis",fontsize=int(self.font_size))
            self.ax_j.set_xlabel("Real Axis",fontsize=self.big_font)
            self.ax_j.set_ylabel("Imaginary Axis",fontsize=self.big_font)

            self.toolbar1 = NavigationToolbar2Tk(self.canvas_j, window,pack_toolbar=False)
            self.toolbar2 = NavigationToolbar2Tk(self.canvas_m, entry_frame1,pack_toolbar=False)
        self.start_switch = True
        self.start_task()

    def create_cmap_dropdowns(self):
        cmap1_label = Label(entry_frame1, text="Outside colormap:", font=self.basic_font)
        cmap2_label = Label(entry_frame1, text="Inside colormap:", font=self.basic_font)

        point_color_label = Label(entry_frame1, text="Points color:", font=self.basic_font)
        line_color_label = Label(entry_frame1, text="Lines color:", font=self.basic_font)

        jump_label = Label(entry_frame1, text="Jump to:", font=self.basic_font)

        self.cmap_options1 = ["plasma", "inferno","viridis", "magma", "cividis","spring","summer",
                              "autumn","winter","hsv", "jet", "coolwarm", "red", "blue", "green", "black", "white"]

        self.cmap_options2 = ["red", "blue", "green", "black", "white"]
        self.jump_options = ["Home", "Flower","Julia island","Seahorse valley","Starfish","Sun","Tendrils","Tree"]

        # Create Combobox for cmap1
        self.cmap1_var = StringVar()
        self.cmap1_var.set(self.cmap1_init)  # Default value
        cmap1_combobox = Combobox(entry_frame1, textvariable=self.cmap1_var, 
                                  values=self.cmap_options1, state="readonly",
                                  font=self.basic_font
                                  )

        # Create Combobox for cmap2
        self.cmap2_var = StringVar()
        self.cmap2_var.set(self.cmap2_init)  # Default value
        cmap2_combobox = Combobox(entry_frame1, textvariable=self.cmap2_var, 
                                  values=self.cmap_options1, state="readonly",
                                  font=self.basic_font
                                  )

        # Create Combobox for point color
        self.point_color_var = StringVar()
        self.point_color_var.set("red")  # Default value
        point_color_combobox = Combobox(entry_frame1, textvariable=self.point_color_var, 
                                        values=self.cmap_options2, state="readonly",
                                        font=self.basic_font
                                        )

        # Create Combobox for line color
        self.line_color_var = StringVar()
        self.line_color_var.set("blue")  # Default value
        line_color_combobox = Combobox(entry_frame1, textvariable=self.line_color_var, 
                                       values=self.cmap_options2, state="readonly",
                                       font=self.basic_font
                                       )
        
        # Create Combobox for jumping
        self.jump_var = StringVar()
        self.jump_var.set("Home")  # Default value
        jump_combobox = Combobox(entry_frame1, textvariable=self.jump_var, 
                                       values=self.jump_options, state="readonly",
                                       font=self.basic_font
                                       )

        # Grid layout for dropdowns
        cmap1_label.grid(row=6, column=0, padx=10, pady=5, sticky='w', in_=tab1)
        cmap1_combobox.grid(row=6, column=0, padx=180, pady=5, in_=tab1)
        cmap2_label.grid(row=7, column=0, padx=10, pady=5, sticky='w', in_=tab1)
        cmap2_combobox.grid(row=7, column=0, padx=180, pady=5, in_=tab1)

        point_color_label.grid(row=6, column=0, padx=10, pady=5, sticky='e', in_=tab2)
        point_color_combobox.grid(row=6, column=1, padx=10, pady=5, in_=tab2,sticky='w')
        line_color_label.grid(row=7, column=0, padx=10, pady=5, sticky='e', in_=tab2)
        line_color_combobox.grid(row=7, column=1, padx=10, pady=5, in_=tab2,sticky='w')

        jump_label.grid(row=5, column=0, padx=10, pady=5, sticky='w', in_=tab1)
        jump_combobox.grid(row=5, column=0, padx=180, pady=5, in_=tab1)

        # Bind functions to Combobox changes
        cmap1_combobox.bind("<<ComboboxSelected>>", self.update_colormaps)
        cmap2_combobox.bind("<<ComboboxSelected>>", self.update_colormaps)
        point_color_combobox.bind("<<ComboboxSelected>>", self.update_point_color)
        line_color_combobox.bind("<<ComboboxSelected>>", self.update_line_color)
        jump_combobox.bind("<<ComboboxSelected>>", self.jump)
        self.cmap1_combobox = cmap1_combobox
        self.cmap2_combobox = cmap2_combobox
        self.point_color_combobox = point_color_combobox
        self.line_color_combobox = line_color_combobox
        self.jump_combobox = jump_combobox

    def on_zoom_m(self,event):
        if self.thread_toggle == True:
            try:
                new_xlim = event.inaxes.get_xlim()
                new_ylim = event.inaxes.get_ylim()
                self.xmin_entry.delete(0, "end")
                self.xmax_entry.delete(0, "end")
                self.ymin_entry.delete(0, "end")
                self.ymax_entry.delete(0, "end")

                self.xmin_entry.insert(0,str(new_xlim[0]))
                self.xmax_entry.insert(0,str(new_xlim[1]))
                self.ymin_entry.insert(0,str(new_ylim[0]))
                self.ymax_entry.insert(0,str(new_ylim[1]))
            except:
                pass

    def on_zoom_j(self,event):
        if self.thread_toggle == True:
            try:
                self.xlim_j = event.inaxes.get_xlim()
                self.ylim_j = event.inaxes.get_ylim()
            except:
                pass

    def jump(self,*args):
        self.exponent = 2
        if self.jump_var.get() == "Home":
            new_num_points = 600
            new_Nb = 600
            new_xlim = [-1.5,0.5]
            new_ylim = [-1,1]
        elif self.jump_var.get() == "Flower":
            new_num_points = 600
            new_Nb = 90
            new_xlim = [-1.99998588117-3e-9,-1.99998588117+3e-9]
            new_ylim = [-3e-9,3e-9]
        elif self.jump_var.get() == "Julia island":
            new_num_points = 600
            new_Nb = 1200
            new_xlim = [-1.768778804-5e-7,-1.768778804+5e-7]
            new_ylim = [-0.001738911-5e-7,-0.001738911+5e-7]
        elif self.jump_var.get() == "Seahorse valley":
            new_num_points = 600
            new_Nb = 1000
            new_xlim = [-0.743517833-6.5e-3,-0.743517833+6.5e-3]
            new_ylim = [0.1270945775-6.5e-3,0.1270945775+6.5e-3]
        elif self.jump_var.get() == "Starfish":
            new_num_points = 600
            new_Nb = 600
            new_xlim = [-0.3739735-7e-5,-0.3739735+7e-5]
            new_ylim = [-0.65977-7e-5,-0.65977+7e-5]
        elif self.jump_var.get() == "Sun":
            new_num_points = 600
            new_Nb = 600
            new_xlim = [-0.776592-9e-5,-0.776592+9e-5]
            new_ylim = [0.136642-9e-5,0.136642+9e-5]
        elif self.jump_var.get() == "Tendrils":
            new_num_points = 600
            new_Nb = 300
            new_xlim = [-0.226266648-1e-6,-0.226266648+1e-6]
            new_ylim = [-1.11617444-1e-6,-1.11617444+1e-6]
        else:
            new_num_points = 600
            new_Nb = 100
            new_xlim = [-1.940157343-1e-6,-1.940157343+1e-6]
            new_ylim = [8e-7-1e-6,8e-7+1e-6]

        self.exponent_entry.delete(0, "end")
        self.num_points_entry.delete(0, "end")
        self.Nb_entry.delete(0, "end")
        self.xmin_entry.delete(0, "end")
        self.xmax_entry.delete(0, "end")
        self.ymin_entry.delete(0, "end")
        self.ymax_entry.delete(0, "end")

        self.exponent_entry.insert(0,str(2))
        self.num_points_entry.insert(0,str(new_num_points))
        self.Nb_entry.insert(0,str(new_Nb))
        self.xmin_entry.insert(0,str(new_xlim[0]))
        self.xmax_entry.insert(0,str(new_xlim[1]))
        self.ymin_entry.insert(0,str(new_ylim[0]))
        self.ymax_entry.insert(0,str(new_ylim[1]))

        self.start_jump = True
        self.start_task()

    def update_point_color(self, *args):
        # Get the selected colormaps from the dropdown menus
        selected_point_color = self.point_color_var.get()
        # Update the class variables with the selected colormaps
        self.selected_point_color = selected_point_color
        self.scatter.remove()
        self.scatter_c.remove()
        if self.points_toggle:
            self.scatter = self.ax.scatter(
                self.real_parts, self.imaginary_parts,
                  s=self.point_size, color=self.selected_point_color
                  )
            self.scatter_c = self.ax.scatter(
                self.real_parts[1], self.imaginary_parts[1],
                  s=10*self.point_size, color=self.selected_point_color
                  )
        self.fig = self.ax.get_figure()
        self.canvas_m.draw()

    def update_line_color(self, *args):
        # Get the selected colormaps from the dropdown menus
        selected_line_color = self.line_color_var.get()
        # Update the class variables with the selected colormaps
        self.selected_line_color = selected_line_color
        self.line.remove()
        self.line, = self.ax.plot(
                self.real_parts, self.imaginary_parts,
                linewidth=self.line_width, color=self.selected_line_color
                )
        self.fig = self.ax.get_figure()
        self.canvas_m.draw()
    
    def update_colormaps(self, *args): 
        self.t = time()
        # Get the selected colormaps from the dropdown menus
        selected_cmap1 = self.cmap1_var.get()
        selected_cmap2 = self.cmap2_var.get()

        # Update the class variables with the selected colormaps
        self.selected_cmap1 = selected_cmap1
        self.selected_cmap2 = selected_cmap2

        xmin_Value = self.xmin_entry.get()
        self.xmin = float(xmin_Value) if xmin_Value else self.xmin
        xmax_Value = self.xmax_entry.get()
        self.xmax = float(xmax_Value) if xmax_Value else self.xmax

        ymin_Value = self.ymin_entry.get()
        self.ymin = float(ymin_Value) if ymin_Value else self.ymin
        ymax_Value = self.ymax_entry.get()
        self.ymax = float(ymax_Value) if ymax_Value else self.ymax 

        num_points_Value = self.num_points_entry.get()
        self.num_points = int(num_points_Value) if num_points_Value else self.num_points

        Nb_Value = self.Nb_entry.get()
        self.Nb = int(Nb_Value) if Nb_Value else self.Nb

        exponent_Value = self.exponent_entry.get()
        self.exponent = float(exponent_Value) if exponent_Value else self.exponent

        # Update the plot with the new colormaps
        self.start_colors = True
        self.start_task()

    def update_plot(self):
        self.z = mandelbrot(self.exponent, self.z0, self.c, self.N, self.ax.get_xlim(), self.ax.get_ylim())
        self.real_parts = [z_i.real for z_i in self.z]
        self.imaginary_parts = [z_i.imag for z_i in self.z]
        if self.lines_toggle:
            self.line.set_xdata(self.real_parts)
            self.line.set_ydata(self.imaginary_parts)
        if self.points_toggle:
            self.scatter.set_offsets(list(zip(self.real_parts, self.imaginary_parts)))
        self.fig = self.ax.get_figure()
        self.canvas_m.draw()
        self.update_result_label()
        self.update_function_label()

    def update_points(self):
        self.points_toggle = not self.points_toggle
        if self.points_toggle:
            self.toggle_points_button.config(relief = RAISED)
            self.scatter = self.ax.scatter(
                self.real_parts, self.imaginary_parts,
                  s=self.point_size,
                  color=mcolors.to_rgba(self.point_color_var.get())
                  )
            self.scatter_c = self.ax.scatter(
                self.real_parts[1], self.imaginary_parts[1],
                  s=10*self.point_size,
                  color=mcolors.to_rgba(self.point_color_var.get())
                  )
            self.point_color_combobox.config(state="readonly")
        else:
            self.toggle_points_button.config(relief = SUNKEN)
            self.point_color_combobox.config(state="disabled")
            self.scatter.remove()  # Remove the scatter plot
            self.scatter_c.remove()
        self.fig = self.ax.get_figure()
        self.canvas_m.draw()

    def update_lines(self):
        self.lines_toggle = not self.lines_toggle
        if self.lines_toggle:
            self.toggle_lines_button.config(relief = RAISED)
            self.line, = self.ax.plot(
                self.real_parts, self.imaginary_parts,
                  linewidth=self.line_width,
                  color=mcolors.to_rgba(self.line_color_var.get())
                  )
            self.line_color_combobox.config(state="readonly")
        else:
            self.toggle_lines_button.config(relief = SUNKEN)
            self.line_color_combobox.config(state="disabled")
            self.line.remove()  # Remove the lines
        self.fig = self.ax.get_figure()
        self.canvas_m.draw()

    def update_result_label(self):
        exponent_Value = self.exponent_entry.get()
        self.exponent = float(exponent_Value) if exponent_Value else self.exponent
        if self.exponent == 2:
            name = "Mandelbrot set"
        elif self.exponent < 2:
            name = "set"
            self.ax.set_title("",fontsize=int(2*self.font_size))
        else:
            name = "Multibrot set"
        if isinstance(self.z, list) and abs(self.z[-1]) > 2:
            self.result_label.config(
                text=f"The selected number {round(self.c,5)}\n is not in the {name}",
                fg="red"
                )
        else:
            self.result_label.config(
                text=f"The selected number {round(self.c,5)}\n is in the {name}",
                fg="black"
                )
            
    def update_function_label(self):
        try:
            exponent_value = self.exponent_entry.get()
            exponent = float(exponent_value) if exponent_value else self.exponent
            exp_dict = [u"\u2070", u"\u00B9", u"\u00B2", u"\u00B3", u"\u2074", u"\u2075", u"\u2076", u"\u2077", u"\u2078", u"\u2079"]

            # Check if the exponent is an integer or a float
            if exponent.is_integer():
                digits = [int(digit) for digit in str(int(exponent))]
                decimal_digits = []
            else:
                # If the exponent is a float, convert it to a string and handle each digit separately
                int_part, dec_part = map(int, str(exponent).split('.'))
                int_digits = [int(digit) for digit in str(int_part)]
                decimal_digits = [int(digit) for digit in str(dec_part)]
                digits = int_digits + decimal_digits

            exp_str = [exp_dict[digit] for digit in digits]

            if exponent == 1:
                exp_str = ""

            # Insert decimal point if there are decimal digits
            if decimal_digits:
                exp_str.insert(len(int_digits), '\uA78F')

            self.function_label.config(
                text=u"Function definition: z\u2099\u208A\u2081 = z" + ''.join(exp_str) + "\u2099 + c"
            )
        except:
            pass

    def clear_entries(self):
        entry_data = [
            (self.N_entry, 'N'),
            (self.point_size_entry, 'point_size'),
            (self.line_width_entry, 'line_width'),
            (self.Nb_entry, 'Nb'),
            (self.num_points_entry, 'num_points'),
            (self.real_entry, 'real_temp'),
            (self.imag_entry, 'imag_temp'),
            (self.exponent_entry, 'exponent'),
            (self.xmin_entry, 'xmin'),
            (self.xmax_entry, 'xmax'),
            (self.ymin_entry, 'ymin'),
            (self.ymax_entry, 'ymax'),
            (self.num_points_j_entry, 'num_points_j'),
            (self.Nb_j_entry, 'Nb_j'),
            (self.zoom_entry, 'zoom_factor')
        ]

        for entry, attribute in entry_data:
            try:
                value = float(entry.get()) if '.' in entry.get() else int(entry.get())
                setattr(self, attribute, value)
            except ValueError:
                pass
            finally:
                entry.delete(0, "end")

    def on_click(self, event):
        self.t = time()
        if event.button == 1 and self.thread_toggle:
            if event.xdata == None:
                pass
            else:
                x, y = event.xdata, event.ydata
                self.real_entry.delete(0, "end")
                self.imag_entry.delete(0, "end")

                self.real_entry.insert(0,str(round(x,10)))
                self.imag_entry.insert(0,str(round(y,10)))

                self.c = complex(x, y)
                self.start_on_click = True
                self.start_task()

    def run_mandelbrot(self,*args):
        self.t = time()
        self.progress_bar.lift()
        self.loaded_label.config(text=f"Working...", fg = "orange")
        try:
            self.point_size_value = self.point_size_entry.get()
            self.point_size = int(self.point_size_value) if self.point_size_value else self.point_size

            self.line_width_value = self.line_width_entry.get()
            self.line_width = float(self.line_width_value) if self.line_width_value else self.line_width

            self.N_value = self.N_entry.get()
            self.N = int(self.N_value) if self.N_value else self.N

            self.real_part_value = self.real_entry.get()
            self.real_part_value = float(self.real_part_value) if self.real_part_value else self.real_temp

            self.imag_part_value = self.imag_entry.get()
            self.imag_part_value = float(self.imag_part_value) if self.imag_part_value else self.imag_temp
            self.exponent_value = self.exponent_entry.get()
            self.exponent = float(self.exponent_value) if self.exponent_value else self.exponent
            self.num_points_Value = self.num_points_entry.get()
            self.num_points = int(self.num_points_Value) if self.num_points_Value else self.num_points

            self.Nb_Value = self.Nb_entry.get()
            self.Nb = int(self.Nb_Value) if self.Nb_Value else self.Nb

            self.xmin_Value = self.xmin_entry.get()
            self.xmin = float(self.xmin_Value) if self.xmin_Value else self.xmin

            self.xmax_Value = self.xmax_entry.get()
            self.xmax = float(self.xmax_Value) if self.xmax_Value else self.xmax

            self.ymin_Value = self.ymin_entry.get()
            self.ymin = float(self.ymin_Value) if self.ymin_Value else self.ymin

            self.ymax_value = self.ymax_entry.get()
            self.ymax = float(self.ymax_value) if self.ymax_value else self.ymax

            zoom_value = self.zoom_entry.get()
            self.zoom_factor = float(zoom_value) if zoom_value else self.zoom_factor

            if self.N> self.N_scale['to']:
                self.N_scale.configure(to=self.N)
            self.N_scale.set(self.N)

            if self.Nb> self.Nb_scale['to']:
                self.Nb_scale.configure(to=self.Nb)
            self.Nb_scale.set(self.Nb)

            if self.point_size> self.point_scale['to']:
                self.point_scale.configure(to=self.point_size)
            self.point_scale.set(self.point_size)

            if self.line_width> self.line_scale['to']:
                self.line_scale.configure(to=self.line_width)
            self.line_scale.set(self.line_width)

            if self.exponent> self.exponent_scale['to']:
                self.exponent_scale.configure(to=self.exponent)
            self.exponent_scale.set(self.exponent)

            if self.num_points> self.num_points_scale['to']:
                self.num_points_scale.configure(to=self.num_points)
            self.num_points_scale.set(self.num_points)

            if self.zoom_factor> self.zoom_scale['to']:
                self.zoom_scale.configure(to=self.zoom_factor)
            self.zoom_scale.set(self.zoom_factor)

            if self.xmin == -1.5 and self.xmax == 0.5 and self.ymin == -1.0 and self.ymax == 1.0 and self.exponent != 2:
                self.xmin = -1.5
                self.xmax = 1.5
                self.ymin = -1.5
                self.ymax = 1.5

                self.xmin_entry.delete(0, "end")
                self.xmax_entry.delete(0, "end")
                self.ymin_entry.delete(0, "end")
                self.ymax_entry.delete(0, "end")

                self.xmin_entry.insert(0,str(-1.5))
                self.xmax_entry.insert(0,str(1.5))
                self.ymin_entry.insert(0,str(-1.5))
                self.ymax_entry.insert(0,str(1.5))
            if self.xmin == -1.5 and self.xmax == 1.5 and self.ymin == -1.5 and self.ymax == 1.5 and self.exponent == 2:
                self.xmin = -1.5
                self.xmax = 0.5
                self.ymin = -1.0
                self.ymax = 1.0

                self.xmin_entry.delete(0, "end")
                self.xmax_entry.delete(0, "end")
                self.ymin_entry.delete(0, "end")
                self.ymax_entry.delete(0, "end")

                self.xmin_entry.insert(0,str(-1.5))
                self.xmax_entry.insert(0,str(0.5))
                self.ymin_entry.insert(0,str(-1.0))
                self.ymax_entry.insert(0,str(1.0))

            self.colors = zeros((self.num_points, self.num_points, 4))
            self.c = complex(self.real_part_value, self.imag_part_value)

            self.update_function_label()
            self.progress_var.set(10)
   
            if self.start_exponent_entry or self.run_task_button or self.start_point_color or self.start_on_click or self.start_switch or self.start_point_position:
                self.update_plot()
            self.progress_var.set(20)

            if self.start_exponent_entry or self.run_task_button or self.start_colors or self.start_jump or self.start_mandelbrot_entry or self.start_switch or self.start_zoom:
                self.update_plot_b()
            self.progress_var.set(30)

            if self.julia_toggle:
                if self.start_exponent_entry or self.run_task_button or self.start_colors or self.start_julia_entry or self.start_on_click or self.start_zoom_j or self.start_switch or self.start_point_position:
                    self.update_julia()
            self.progress_var.set(40)

            if self.lines_toggle:
                self.update_line_color()
            self.progress_var.set(50)

            if self.points_toggle:
                self.update_point_color()
            self.progress_var.set(60)

            self.update_loaded_label()
            self.toolbar1.place(x = 0, y = 50)
            self.toolbar1.update()
            self.progress_var.set(70)

            self.toolbar2.place(x = 0, y = 447)
            self.toolbar2.update()
            self.progress_var.set(80)

            self.toolbar1.lift()
            self.progress_var.set(90)

            self.toolbar2.lift()
            self.progress_var.set(100)
        except ValueError:
            
            messagebox.showwarning("Invalid Input", "Please enter a valid value.\nOnly positive integers or floats are allowed")
            self.update_loaded_label()
            self.progress_var.set(100)
        self.start_on_click = False
        self.start_colors = False
        self.start_point_color = False
        self.start_point_position = False
        self.start_julia_entry = False
        self.start_mandelbrot_entry = False
        self.start_jump = False
        self.start_switch = False
        self.start_zoom = False
        self.start_zoom_j = False
        self.run_task_button = False
        self.start_exponent_entry = False

        self.thread_toggle = True

    def update_loaded_label(self):
        self.loaded_label.config(text="Loaded", fg = "lime")
        elapsed = time() - self.t
        window.after(500, lambda: self.loaded_label.config(text=f"Idle, executed in {round(elapsed,4)}s", fg = "blue"))
        self.loaded_label.lift()
        
    def extract_rgba_values(self,N, ax, canvas, cmap, ticks, ticklabels):
        arr = linspace(0, N, N).reshape((N, -1))
        rotated_arr = rot90(arr)
        img = ax.imshow(rotated_arr, interpolation='nearest', cmap=cmap, aspect='auto')
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels)
        canvas.draw()
        rgba_values = []
        for i in range(N):
            color = img.cmap(i / (N - 1))
            rgba_values.append(color)
        return array(rgba_values)

    def extract_rgba_values1(self):
        return self.extract_rgba_values(self.Nb, self.ax1, self.canvas1, self.cmap1, [0, self.Nb//2, self.Nb-1], [0, self.Nb//2, self.Nb])

    def extract_rgba_values2(self):
        return self.extract_rgba_values(self.Nb, self.ax2, self.canvas2, self.cmap2, [0, self.Nb//2, self.Nb-1], [0, 1.5, 3])

    def extract_rgba_values3(self):
        return self.extract_rgba_values(self.Nb_j, self.ax3, self.canvas3, self.cmap1, [0, self.Nb_j//2, self.Nb_j-1], [0, self.Nb_j//2, self.Nb_j])

    def extract_rgba_values4(self):
        return self.extract_rgba_values(self.Nb_j, self.ax4, self.canvas4, self.cmap2, [0, self.Nb_j//2, self.Nb_j-1], [0, 1.5, 3])

    def update_plot_b(self):
        x = linspace(self.xmin, self.xmax, self.num_points)
        y = linspace(self.ymin, self.ymax, self.num_points)
        X, Y = meshgrid(x, y)
        complex_grid = X + 1j * Y
        colors = empty((self.num_points, self.num_points, 4))
            
        if self.cmap1_var.get() in plt.colormaps():
            self.cmap1 = plt.get_cmap(self.cmap1_var.get())
            cmap1 = array(self.extract_rgba_values1())
            color_toggle1 = True
        else:
            color_toggle1 = False
            cmap1 = array([mcolors.to_rgba(self.cmap1_var.get())])
            
        if self.cmap2_var.get() in plt.colormaps():
            self.cmap2 = plt.get_cmap(self.cmap2_var.get())
            cmap2 = array(self.extract_rgba_values2())
            color_toggle2 = True
        else:
            color_toggle2 = False
            cmap2 = array([mcolors.to_rgba(self.cmap2_var.get())])

        scale_factor = self.Nb/2
        cmap1_default = cmap1[0, :]
        cmap2_default = cmap2[0, :]

        colors = update_b_core(self.num_points,complex_grid,self.exponent, self.z0, self.Nb,colors,color_toggle1,color_toggle2,cmap1,cmap2,scale_factor,cmap1_default,cmap2_default) 
                        
        self.ax.imshow(colors, extent=[self.xmin, self.xmax, self.ymin, self.ymax], origin='lower')
        self.ax.set_xlim(self.xmin, self.xmax)
        self.ax.set_ylim(self.ymin, self.ymax)
        self.canvas_m.draw()

    def update_julia(self):
        try:
            self.Nb_j_value = self.Nb_j_entry.get()
            self.Nb_j = int(self.Nb_j_value) if self.Nb_j_value else self.Nb_j

            self.num_points_j_value = self.num_points_j_entry.get()
            self.num_points_j = int(self.num_points_j_value) if self.num_points_j_value else self.num_points_j

            self.real_value = self.real_entry.get()
            self.real = float(self.real_value) if self.real_value else self.real

            self.imag_value = self.imag_entry.get()
            self.imag = float(self.imag_value) if self.imag_value else self.imag
            self.c = complex(self.real, self.imag)

            if self.num_points_j> self.num_points_j_scale['to']:
                self.num_points_j_scale.configure(to=self.num_points_j)
            self.num_points_j_scale.set(self.num_points_j)

            if self.Nb_j> self.Nb_j_scale['to']:
                self.Nb_j_scale.configure(to=self.Nb_j)
            self.Nb_j_scale.set(self.Nb_j)

            self.exponent_value = self.exponent_entry.get()
            self.exponent = float(self.exponent_value) if self.exponent_value else self.exponent
            self.generate_julia()
        except ValueError:
            messagebox.showwarning("Invalid Input", "Please enter a valid value.\nOnly positive integers or floats are allowed")

    def generate_julia(self):
        x = linspace(self.xlim_j[0], self.xlim_j[1], self.num_points_j)
        y = linspace(self.ylim_j[0], self.ylim_j[1], self.num_points_j)

        X, Y = meshgrid(x, y)
        complex_grid = X + 1j * Y
        colors = zeros((self.num_points_j, self.num_points_j, 4))

        if not self.julia_toggle:
            cmap1 = self.cmap1_temp
            cmap2 = self.cmap2_temp
        else:
            cmap1 = (self.cmap1_var.get())
            cmap2 = (self.cmap2_var.get())

        if self.cmap1_var.get() in plt.colormaps():
            self.cmap1 = plt.get_cmap(self.cmap1_var.get())
            cmap1 = array(self.extract_rgba_values3())
            color_toggle1 = True
        else:
            color_toggle1 = False
            cmap1 = array([mcolors.to_rgba(self.cmap1_var.get())])
            
        if self.cmap2_var.get() in plt.colormaps():
            self.cmap2 = plt.get_cmap(self.cmap2_var.get())
            cmap2 = array(self.extract_rgba_values4())
            color_toggle2 = True
        else:
            color_toggle2 = False
            cmap2 = array([mcolors.to_rgba(self.cmap2_var.get())])

        scale_factor = self.Nb_j/2
        cmap1_default = cmap1[0, :]
        cmap2_default = cmap2[0, :]
        colors = update_julia_core(self.c,self.num_points_j,complex_grid,self.exponent, self.z0, self.Nb_j,colors,color_toggle1,color_toggle2,cmap1,cmap2,scale_factor,cmap1_default,cmap2_default) 
         
        self.ax_j.imshow(colors, extent=[self.xlim_j[0], self.xlim_j[1], 
                                         self.ylim_j[0], self.ylim_j[1]], origin='lower')
        self.ax_j.set_xlim(self.xlim_j[0], self.xlim_j[1])
        self.ax_j.set_ylim(self.ylim_j[0], self.ylim_j[1])
        self.canvas_j.draw()
        
    def control_shortcuts(self, *args):
        self.shortcuts_toggle = not self.shortcuts_toggle
        actions = {
            '<Control-e>': self.run_task,
            '<Control-d>': self.default_entries,
            '<Control-z>': self.run_zoom_entry,
            '<Control-w>': self.switch_plots,
            '<Control-n>': self.new_window,
            '<Control-r>': self.restart,
            '<Control-m>': self.save_m,
            '<Control-j>': self.save_j,
            '<Control-h>': self.hint,
            '<Control-g>': self.about
        }

        keys_to_unbind = ['<e>', '<d>', '<z>', '<w>', '<n>', '<r>', '<m>', '<j>', '<h>', '<g>']

        if self.shortcuts_toggle:
            self.shortcut_label.config(text="")
        else:
            self.shortcut_label.config(text="Simple shortcuts\nenabled!")

        for key in keys_to_unbind:
            self.window.unbind(key)

        for key, action in actions.items():
            bind_key = key if self.shortcuts_toggle else key.replace('<Control-', '<')
            self.window.bind(bind_key, action)


    def hint(self,*args):
        message=('You can update the entered values directly by pressing enter, otherwise press the "Run" button.'
                 ' Note that the "Run" button attempts to update all values even if not needed.\n'
                 'By selecting the zoom factor to be eg. 0.1, the "Zoom to point" button will instead zoom out.\n'
                 'You can select the coordinates of point c by clicking the plot.\n'
                 'Note that selecting a larger grid size will cause the rendering to take longer.\n'
                 'Scales and combo boxes are scrollable.\n\n'
                 'Please wait for the plots to finish loading before you enter inputs.')
        messagebox.showinfo("Hint",message)
        
    def about(self, *args):
        # Check if TopLevel window is already present
        if not hasattr(self, 'top_about') or not self.top_about.winfo_exists():
            # Create a Toplevel window
            self.top_about = Toplevel(window)
            self.top_about.title("About")

            # Calculate the position to center the window on the screen
            screen_width = self.top_about.winfo_screenwidth()
            screen_height = self.top_about.winfo_screenheight()
            window_width = 600  # Set the width of your window
            window_height = 300  # Set the height of your window

            x_position = (screen_width - window_width) // 2
            y_position = (screen_height - window_height) // 2

            # Set the geometry of the window to center it
            self.top_about.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

            # Link1
            link1_label = Label(self.top_about, text='You can find more about the Mandelbrot set here:',
                                font=('Arial', self.font_size))
            link1_label.grid(column=0, row=0, padx=10, pady=10, sticky="nw")

            link1 = Label(self.top_about, text='https://en.wikipedia.org/wiki/Mandelbrot_set', font=self.link_font,
                        fg="blue", cursor="hand2")
            link1.grid(column=0, row=1, padx=10, pady=10, sticky="nw")
            link1.bind("<Button-1>", lambda event: self.openbrowser('https://en.wikipedia.org/wiki/Mandelbrot_set'))

            # Link2
            link2_label = Label(self.top_about, text='You can find more about the Julia set here:',
                                font=('Arial', self.font_size))
            link2_label.grid(column=0, row=3, padx=10, pady=10, sticky="nw")

            link2 = Label(self.top_about, text='https://en.wikipedia.org/wiki/Julia_set', font=self.link_font,
                        fg="blue", cursor="hand2")
            link2.grid(column=0, row=4, padx=10, pady=10, sticky="nw")
            link2.bind("<Button-1>", lambda event: self.openbrowser('https://en.wikipedia.org/wiki/Julia_set'))

    def openbrowser(self, url):
        webbrowser.open_new(url)

    def shortcuts(self,*args):
        message=(   
                'Ctrl + E - Updates all parameters\n'
                'Ctrl + D - Sets the values to default\n'
                'Ctrl + Z - Zooms to placed point \n'
                'Ctrl + W - Switches plots\n'
                'Ctrl + N - Opens new window\n'
                'Ctrl + R - Restarts the app\n'
                'Ctrl + M - Brings the Mandelbrot set save dialog\n'
                'Ctrl + J - Brings the Julia set save dialog\n'
                'Ctrl + H - Shows the "Hint" message\n'
                'Ctrl + G - Shows the "About" message\n'
                'F1 - Shows this message\n'
                'Alt + F4 - Closes the app\n\n'
                'Alt + Q - Disables Ctrl in other shortcuts\n(e.g. only pressing "E" will now update all parameters)'
                )
        messagebox.showinfo("List Of Shortcuts", message)

    def save(self):
        filetypes = [
            ("Encapsuleted Postscript", "*.eps"),
            ("Joint Photographic Experts Group", "*.eps;*.jpg"),
            ("PGF code for LaTeX", "*.pgf"),
            ("Portable Document Format", "*.pdf"),
            ("Portable Network Graphics", "*.png"),
            ("Postscript", "*.ps"),
            ("Raw RGBA bitmap", "*.raw;*.rgba"),
            ("Scalable Vector Graphics", "*.svg;*.svgz"),
            ("Tagged Image File Format", "*.tif;*.tiff"),
            ("WebP Image Format", "*.webp")]

        self.fw = filedialog.asksaveasfile(mode='w', defaultextension=".png", filetypes=filetypes)

        if self.fw is None:
            return
        
    def save_m(self,*args):
        self.save()
        if self.fw is None:
            return
        
        self.fig = self.ax.get_figure()
        self.fig.savefig(self.fw.name, bbox_inches='tight')
        self.fw.close()
        messagebox.showinfo(title="File was created", message="Success!")

    def save_j(self,*args):
        self.save()
        if self.fw is None:
            return
        
        self.fig = self.ax_j.get_figure()
        self.fig.savefig(self.fw.name, bbox_inches='tight')
        self.fw.close()
        messagebox.showinfo(title="File was created", message="Success!")

# Create a GUI window
window = Tk()
window.title("Mandelbrot Set Calculator")
window.state("zoomed")

# Get the current working directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Load the image from the current directory
image_path = os.path.join(current_directory, 'm_icon.png')
image = Image.open(image_path)
photo = ImageTk.PhotoImage(image)
window.iconphoto(True, photo)

# Create a frame to hold the entry widgets
entry_frame1 = Frame(master=window)
entry_frame1.grid(row=0, column=1, padx=10, sticky="ne")

# Create a Notebook widget 
entry_tabs = Notebook(entry_frame1)
entry_tabs.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="e")

# Create the first tab 
tab1 = Frame(entry_tabs)
entry_tabs.add(tab1, text="Main controls")

# Create the second tab 
tab2 = Frame(entry_tabs)
entry_tabs.add(tab2, text="Point attributes")

# Create the third tab 
tab3 = Frame(entry_tabs)
entry_tabs.add(tab3, text="Background attributes")

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

# Create an instance of MandelbrotPlotter
mandelbrot_plotter = MandelbrotPlotter(window, entry_frame1)

# Start the GUI event loop
window.mainloop()