"""gbhobbytech colours and application-wide ttk styling; no service logic."""
from __future__ import annotations

import re

DEFAULT_APPEARANCE = {
    'mode': 'light',
    'primary': '#16324F',
    'accent': '#B8734F',
    'font_size': 10,
}


def validate_appearance(raw) -> dict:
    if not isinstance(raw, dict):
        raise ValueError('Appearance settings must be a table')
    values = {**DEFAULT_APPEARANCE, **raw}
    if values['mode'] not in ('light', 'dark'):
        raise ValueError('Appearance mode must be light or dark')
    for key in ('primary', 'accent'):
        if not isinstance(values[key], str) or not re.fullmatch(r'#[0-9a-fA-F]{6}', values[key]):
            raise ValueError(f'{key.title()} colour must be a six-digit hex colour, such as #16324F')
    if isinstance(values['font_size'], bool) or not isinstance(values['font_size'], int) or not 9 <= values['font_size'] <= 14:
        raise ValueError('Text size must be a whole number from 9 to 14')
    return {key: values[key] for key in DEFAULT_APPEARANCE}


def contrast_text(colour: str) -> str:
    values = [int(colour[i:i+2], 16) / 255 for i in (1, 3, 5)]
    r, g, b = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4 for v in values]
    luminance = .2126*r + .7152*g + .0722*b
    return '#111111' if (luminance + .05) / .055 >= 1.05 / (luminance + .05) else '#FFFFFF'


def apply_theme(root, values=None) -> dict:
    import tkinter as tk
    from tkinter import font, ttk
    values = validate_appearance(values or {})
    dark = values['mode'] == 'dark'
    palette = dict(values, background='#141D27' if dark else '#EDF1F5',
                   panel='#202D3B' if dark else '#FFFFFF',
                   field='#16212D' if dark else '#F7F9FB',
                   text='#EEF2F6' if dark else '#293440',
                   muted='#B7C4D0' if dark else '#546577',
                   border='#435367' if dark else '#CCD6DF',
                   ready='#8CE0AC' if dark else '#19683B',
                   warning='#F2CB7D' if dark else '#82540B',
                   error='#FFABB4' if dark else '#A32940')
    root._theme = palette
    style = ttk.Style(root)
    style.theme_use('clam')
    families = set(font.families(root))
    family = 'Aptos' if 'Aptos' in families else ('DejaVu Sans' if 'DejaVu Sans' in families else font.nametofont('TkDefaultFont').cget('family'))
    size = values['font_size']
    for name in ('TkDefaultFont', 'TkTextFont', 'TkMenuFont', 'TkHeadingFont'):
        font.nametofont(name).configure(family=family, size=size)
    # Retain named font objects: deleting Python Font objects deletes the Tcl fonts.
    if not hasattr(root, '_app_fonts'):
        root._app_fonts = {name: font.Font(root=root, name=name) for name in ('AppTitle', 'AppHeading')}
    root._app_fonts['AppTitle'].configure(family=family, size=size+8, weight='bold')
    root._app_fonts['AppHeading'].configure(family=family, size=size+1, weight='bold')
    style.configure('.', background=palette['panel'], foreground=palette['text'], font='TkDefaultFont', bordercolor=palette['border'])
    style.configure('TFrame', background=palette['panel'])
    style.configure('Background.TFrame', background=palette['background'])
    style.configure('TLabel', background=palette['panel'], foreground=palette['text'])
    style.configure('Muted.TLabel', foreground=palette['muted'])
    style.configure('Title.TLabel', font='AppTitle', foreground=palette['text'])
    style.configure('TLabelframe', background=palette['panel'], bordercolor=palette['border'], relief='solid', borderwidth=1)
    style.configure('TLabelframe.Label', font='AppHeading', foreground=palette['text'], background=palette['panel'])
    style.configure('TButton', padding=(12, 7), relief='flat', background=palette['field'], bordercolor=palette['border'])
    style.map('TButton', background=[('pressed', palette['border']), ('active', palette['background'])], foreground=[('disabled', palette['muted'])])
    for name, colour in (('Primary', palette['primary']), ('Accent', palette['accent'])):
        style.configure(name+'.TButton', background=colour, foreground=contrast_text(colour), bordercolor=colour, font='AppHeading')
        style.map(name+'.TButton', background=[('active', colour), ('pressed', colour)], foreground=[('disabled', palette['muted']), ('!disabled', contrast_text(colour))])
    style.configure('Danger.TButton', foreground=palette['error'])
    style.configure('Quiet.TButton', background=palette['panel'], borderwidth=0, foreground=palette['muted'])
    style.configure('Task.TButton', padding=(14, 11), font='AppHeading')
    style.configure('Active.Task.TButton', background=palette['primary'], foreground=contrast_text(palette['primary']))
    style.map('Active.Task.TButton', background=[('active', palette['primary'])], foreground=[('!disabled', contrast_text(palette['primary']))])
    for state, colour in (('Ready', 'ready'), ('Starting', 'warning'), ('Error', 'error'), ('Stopped', 'muted'), ('External', 'muted')):
        style.configure(state+'.TLabel', foreground=palette[colour])
    style.configure('TEntry', fieldbackground=palette['field'], foreground=palette['text'], padding=6)
    style.configure('TMenubutton', background=palette['field'], foreground=palette['text'], padding=(10, 7), relief='flat')
    style.configure('TNotebook', background=palette['background'], borderwidth=0)
    style.configure('TNotebook.Tab', padding=(12, 8), background=palette['background'])
    style.map('TNotebook.Tab', background=[('selected', palette['panel'])], foreground=[('selected', palette['text'])])
    style.configure('Horizontal.TProgressbar', background=palette['accent'], troughcolor=palette['background'], borderwidth=0, thickness=7)
    style.configure('TScrollbar', background=palette['border'], troughcolor=palette['field'], borderwidth=0, arrowsize=12)
    style.map('TCheckbutton', background=[('active', palette['panel'])])
    style.map('TRadiobutton', background=[('active', palette['panel'])])
    root.configure(background=palette['background'])
    style_classic(root)
    return palette


def style_classic(widget) -> None:
    """Theme Text, Listbox and Menu widgets, including newly opened dialogs."""
    import tkinter as tk
    root = widget._root()
    p = getattr(root, '_theme', None)
    if not p:
        return
    if isinstance(widget, (tk.Tk, tk.Toplevel)):
        widget.configure(background=p['background'])
    if isinstance(widget, (tk.Text, tk.Listbox)):
        widget.configure(background=p['field'], foreground=p['text'],
                         selectbackground=p['primary'], selectforeground=contrast_text(p['primary']),
                         highlightbackground=p['border'], highlightcolor=p['accent'],
                         highlightthickness=1, borderwidth=0, relief='flat', font='TkTextFont')
        if isinstance(widget, tk.Text):
            widget.configure(insertbackground=p['text'], padx=10, pady=9)
    elif isinstance(widget, tk.Menu):
        widget.configure(background=p['panel'], foreground=p['text'],
                         activebackground=p['primary'], activeforeground=contrast_text(p['primary']),
                         borderwidth=0, font='TkMenuFont')
    for child in widget.winfo_children():
        style_classic(child)
