import os
import sys
import math
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

import tkinter as tk
from tkinter import ttk, messagebox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# This simple desktop app reads the CSVs produced by stormy_fetch.py
# and turns them into clear visuals (4 tabs) so anyone can quickly
# understand “what it will feel like” today and this week, and how
# the current year compares to 2023/2024.
plt.style.use('dark_background')


APP_TITLE = "Stormy"
DEFAULT_WINDOW_SIZE = (1600, 950)  # wider than 1280x800 as requested
DATA_DIR = "CSVs"  # where CSV outputs are expected by default
# App backgrounds and accents
APP_BG = '#0b0f14'        # main dark background behind plots
CTRL_BG = '#0f172a'       # control bar background (deep slate/blue)
CTRL_TEXT = '#e5e7eb'     # light text on control bar
BTN_BG = '#2563eb'        # action button background
BTN_BG_ACTIVE = '#1d4ed8' # hover/active color
COMBO_BG = '#1e293b'      # combobox field background
COMBO_FG = '#e5e7eb'


def _load_csv(path: str, parse_dates: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, parse_dates=parse_dates)
    except Exception as e:
        print(f"Failed to read {path}: {e}", file=sys.stderr)
        return None


class StormyApp(tk.Tk):
    # The main window: loads data, builds controls & tabs, and refreshes plots
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        # Set window background to match the app theme and align Matplotlib defaults
        self.configure(bg=APP_BG)
        self.geometry(f"{DEFAULT_WINDOW_SIZE[0]}x{DEFAULT_WINDOW_SIZE[1]}")
        try:
            plt.rcParams.update({
                'figure.facecolor': APP_BG,
                'axes.facecolor': APP_BG,
                'savefig.facecolor': APP_BG,
            })
        except Exception:
            pass

        # 1) Load data from CSVs (if files are missing, we show a helpful message)
        _h = _load_csv(os.path.join(DATA_DIR, "hourly.csv"), parse_dates=["datetime_local"]) 
        self.df_hourly = _h if _h is not None else pd.DataFrame()
        _d = _load_csv(os.path.join(DATA_DIR, "daily.csv"), parse_dates=["date_local"]) 
        self.df_daily = _d if _d is not None else pd.DataFrame()
        _hd = _load_csv(os.path.join(DATA_DIR, "historical_daily.csv"), parse_dates=["date_local"]) 
        self.h_daily = _hd if _hd is not None else pd.DataFrame()
        _hm = _load_csv(os.path.join(DATA_DIR, "historical_monthly.csv")) 
        self.h_monthly = _hm if _hm is not None else pd.DataFrame()
        _hq = _load_csv(os.path.join(DATA_DIR, "historical_quarterly.csv")) 
        self.h_quarterly = _hq if _hq is not None else pd.DataFrame()

        if self.df_daily.empty or self.df_hourly.empty:
            messagebox.showerror(
                APP_TITLE,
                "Missing CSVs. Please run 'python stormy_fetch.py' first to generate CSVs in the 'CSVs' folder.",
            )

        # Prepare the city dropdown (sorted list from the daily data)
        self.cities = sorted(self.df_daily["city"].unique().tolist()) if not self.df_daily.empty else []
        self.selected_city = tk.StringVar(value=self.cities[0] if self.cities else "")

        # 2) Controls (city selector + buttons) on a colored bar
        ctl = tk.Frame(self, bg=CTRL_BG, highlightthickness=0, bd=0)
        ctl.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        tk.Label(ctl, text="City:", font=("Segoe UI", 10, "bold"), bg=CTRL_BG, fg=CTRL_TEXT).pack(side=tk.LEFT)
        self.city_cb = ttk.Combobox(ctl, textvariable=self.selected_city, values=self.cities, state="readonly", width=24)
        self.city_cb.pack(side=tk.LEFT, padx=8)
        self.city_cb.bind("<<ComboboxSelected>>", lambda e: self.refresh_all())

        # Small status label showing when the data was last updated (based on hourly.csv)
        self.updated_var = tk.StringVar(value="Updated: –")
        self.updated_label = tk.Label(ctl, textvariable=self.updated_var, bg=CTRL_BG, fg=CTRL_TEXT)
        self.updated_label.pack(side=tk.RIGHT, padx=(8, 0))

        # Quick action buttons
        self.compare_btn = ttk.Button(ctl, text="Compare…", command=self.open_compare_popup)
        self.compare_btn.pack(side=tk.RIGHT, padx=(8, 0))
        self.reload_btn = ttk.Button(ctl, text="Reload CSVs", command=self.reload_data)
        self.reload_btn.pack(side=tk.RIGHT, padx=(8, 0))

        # 3) Tabs for the four views we present in the UI
        # Notebook with a custom style so tabs fill the width and have subtle color/border
        self.style = ttk.Style(self)
        # Use a cross-platform theme that honors tab color settings
        try:
            self.style.theme_use('clam')
        except Exception:
            pass
        # Create a custom style namespace for the notebook and its tabs
        self.style.configure('Stormy.TNotebook', background=APP_BG, tabmargins=[0, 2, 0, 0])
        # Unselected/selected states get a soft background and a thin border for definition
        self.style.configure(
            'Stormy.TNotebook.Tab',
            padding=(12, 8),
            background='#1a1f2b',
            foreground='#e5e7eb',
            borderwidth=1,
            relief='ridge'
        )
        self.style.map(
            'Stormy.TNotebook.Tab',
            background=[('selected', '#263041'), ('active', '#202736')],
            foreground=[('selected', '#ffffff')],
        )
        # Accent styles: colored buttons and a lightly tinted city combobox
        self.style.configure('Action.TButton', background=BTN_BG, foreground='#ffffff', borderwidth=1)
        self.style.map('Action.TButton', background=[('active', BTN_BG_ACTIVE)])
        self.style.configure('City.TCombobox', fieldbackground=COMBO_BG, foreground=COMBO_FG)
        self.style.map('City.TCombobox', fieldbackground=[('active', COMBO_BG), ('readonly', COMBO_BG)])
        self.nb = ttk.Notebook(self, style='Stormy.TNotebook')
        self.nb.pack(expand=True, fill=tk.BOTH)

        # Dark page background for all tabs
        self.style.configure('Dark.TFrame', background=APP_BG)
        self.tab1 = ttk.Frame(self.nb, style='Dark.TFrame')
        self.tab2 = ttk.Frame(self.nb, style='Dark.TFrame')
        self.tab3 = ttk.Frame(self.nb, style='Dark.TFrame')
        self.tab4 = ttk.Frame(self.nb, style='Dark.TFrame')
        self.nb.add(self.tab1, text="Today & Next 7")
        self.nb.add(self.tab2, text="Quarters 2023/2024")
        self.nb.add(self.tab3, text="Differences 23→24")
        self.nb.add(self.tab4, text="Projection")

        # Apply the accent styles to controls created earlier
        try:
            self.compare_btn.configure(style='Action.TButton')
            self.reload_btn.configure(style='Action.TButton')
            self.city_cb.configure(style='City.TCombobox')
        except Exception:
            pass

        # Build the content of each tab
        self._build_tab1()
        self._build_tab2()
        self._build_tab3()
        self._build_tab4()

        # Make tab headers fill the bar equally (each gets 1/N of total width)
        self.nb.bind('<Configure>', self._equalize_tab_widths)
        self.after(0, self._equalize_tab_widths)

        # Update status text and draw everything once at startup
        self._update_last_updated_label()
        self.refresh_all()

    # ---------- Utilities ----------
    def _today_date(self) -> date:
        # Use your computer's local date; our CSVs use local Europe/Bucharest time
        return pd.Timestamp.today().date()

    def _fmt_float(self, x: Optional[float], digits: int = 1, suffix: str = "") -> str:
        # Helper: print numbers like “12.3°C” or “-” if missing
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return "-"
        return f"{x:.{digits}f}{suffix}"

    def _add_note(self, ax, text: str, loc: str = 'upper left'):
        # Tiny legend-like note inside a chart to explain what viewers see
        loc_map = {
            'upper left': (0.01, 0.99, 'left', 'top'),
            'upper right': (0.99, 0.99, 'right', 'top'),
            'lower left': (0.01, 0.01, 'left', 'bottom'),
            'lower right': (0.99, 0.01, 'right', 'bottom'),
        }
        x, y, ha, va = loc_map.get(loc, loc_map['upper left'])
        ax.text(
            x, y, text,
            transform=ax.transAxes,
            ha=ha, va=va,
            fontsize=8,
            color='#dddddd',
            bbox=dict(facecolor=(0, 0, 0, 0.25), edgecolor='none', boxstyle='round,pad=0.3')
        )

    def _update_last_updated_label(self):
        # Show the most recent timestamp from hourly.csv, or a dash if not available
        try:
            if not self.df_hourly.empty and "datetime_local" in self.df_hourly.columns:
                ts = pd.to_datetime(self.df_hourly["datetime_local"], errors="coerce").max()
                if pd.notnull(ts):
                    self.updated_var.set(f"Updated: {ts.strftime('%Y-%m-%d %H:%M')}")
                    return
        except Exception:
            pass
        self.updated_var.set("Updated: –")

    def reload_data(self):
        # Re-read CSV files from disk and refresh views (useful after running the fetcher)
        _h = _load_csv(os.path.join(DATA_DIR, "hourly.csv"), parse_dates=["datetime_local"]) 
        self.df_hourly = _h if _h is not None else pd.DataFrame()
        _d = _load_csv(os.path.join(DATA_DIR, "daily.csv"), parse_dates=["date_local"]) 
        self.df_daily = _d if _d is not None else pd.DataFrame()
        _hd = _load_csv(os.path.join(DATA_DIR, "historical_daily.csv"), parse_dates=["date_local"]) 
        self.h_daily = _hd if _hd is not None else pd.DataFrame()
        _hm = _load_csv(os.path.join(DATA_DIR, "historical_monthly.csv")) 
        self.h_monthly = _hm if _hm is not None else pd.DataFrame()
        _hq = _load_csv(os.path.join(DATA_DIR, "historical_quarterly.csv")) 
        self.h_quarterly = _hq if _hq is not None else pd.DataFrame()

        # Update city list and combobox if needed
        new_cities = sorted(self.df_daily["city"].unique().tolist()) if not self.df_daily.empty else []
        self.city_cb["values"] = new_cities
        if self.selected_city.get() not in new_cities:
            self.selected_city.set(new_cities[0] if new_cities else "")
        self.cities = new_cities

        # Update the status label and re-render views
        self._update_last_updated_label()
        self.refresh_all()

    def _equalize_tab_widths(self, event=None):
        # Ensure notebook tabs share the full width equally.
        # This uses a style width on the Tab element; we recompute on resize.
        try:
            tab_ids = self.nb.tabs()
            n = max(1, len(tab_ids))
            total_w = self.nb.winfo_width()
            # If not yet laid out, try again shortly
            if total_w <= 1:
                self.after(50, self._equalize_tab_widths)
                return
            per = max(100, int(total_w / n))
            # Set a per-tab width via style; many themes honor this for tabs
            self.style.configure('Stormy.TNotebook.Tab', width=per)
        except Exception:
            pass

    # ---------- Tab 1 ----------
    def _build_tab1(self):
        # Tab 1 shows “Today” (hourly detail) on the left
        # and “Next 7 days” (ranges + rain chance) on the right
        # Split in two halves on a dark background
        left = tk.Frame(self.tab1, bg=APP_BG, highlightthickness=0, bd=0)
        right = tk.Frame(self.tab1, bg=APP_BG, highlightthickness=0, bd=0)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 4), pady=8)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 8), pady=8)

        # Left figure: Today hourly temp + rain prob + state ribbon
        self.t1_left_fig = plt.Figure(figsize=(8, 5), constrained_layout=True)
        self.t1_left_fig.patch.set_facecolor(APP_BG)
        gs = self.t1_left_fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3, 1])
        self.t1_ax_main = self.t1_left_fig.add_subplot(gs[0])
        self.t1_ax_state = self.t1_left_fig.add_subplot(gs[1], sharex=self.t1_ax_main)
        for ax in (self.t1_ax_main, self.t1_ax_state):
            try:
                ax.set_facecolor(APP_BG)
            except Exception:
                pass
        self.t1_left_canvas = FigureCanvasTkAgg(self.t1_left_fig, master=left)
        self.t1_left_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # Remove white borders from the canvas
        self.t1_left_canvas.get_tk_widget().configure(bg=APP_BG, highlightthickness=0, bd=0)

        # Right figure: Next 7 days min/max range + mean + rain chance (twin axis)
        self.t1_right_fig, self.t1_right_ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)
        self.t1_right_fig.patch.set_facecolor(APP_BG)
        self.t1_right_ax_rain = self.t1_right_ax.twinx()
        try:
            self.t1_right_ax.set_facecolor(APP_BG)
            self.t1_right_ax_rain.set_facecolor(APP_BG)
        except Exception:
            pass
        self.t1_right_canvas = FigureCanvasTkAgg(self.t1_right_fig, master=right)
        self.t1_right_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.t1_right_canvas.get_tk_widget().configure(bg=APP_BG, highlightthickness=0, bd=0)

    def _refresh_tab1(self):
        # Re-draw Tab 1 using the currently selected city
        city = self.selected_city.get()
        if not city:
            return
        today = self._today_date()

        # Left figure: hourly temperature, rain chance, and a color ribbon for state
        self.t1_ax_main.clear()
        self.t1_ax_state.clear()
        hh = self.df_hourly
        mask_h = (hh["city"] == city) & (pd.to_datetime(hh["datetime_local"]).dt.date == today)
        today_hours = hh.loc[mask_h, ["datetime_local", "temp_c", "precip_prob_pct", "weather_state"]].copy()
        if not today_hours.empty:
            today_hours.sort_values("datetime_local", inplace=True)
            times = pd.to_datetime(today_hours["datetime_local"]) 
            hours = times.dt.hour + times.dt.minute/60.0
            temps = today_hours["temp_c"].astype(float).values
            probs = today_hours["precip_prob_pct"].astype(float).values

            # Lightly shade approximate night hours so the day stands out
            for h0, h1 in [(0, 6), (20, 24)]:
                self.t1_ax_main.axvspan(h0, h1, color='grey', alpha=0.1)

            # Temperature line + a simple 3-hour rolling average (smooths tiny wiggles)
            self.t1_ax_main.plot(hours, temps, color="#ff7f0e", label="Temp (°C)", linewidth=2)
            if len(temps) >= 3:
                roll = pd.Series(temps).rolling(3, min_periods=1).mean().values
                self.t1_ax_main.plot(hours, roll, color="#ffd27f", linestyle="--", linewidth=1.5, label="3h mean")
            self.t1_ax_main.set_xlim(0, 23.99)
            self.t1_ax_main.set_xticks(range(0, 24, 3))
            self.t1_ax_main.set_xlabel("Hour")
            self.t1_ax_main.set_ylabel("°C")
            self.t1_ax_main.grid(True, alpha=0.2)

            ax2 = self.t1_ax_main.twinx()
            ax2.bar(hours, probs, width=0.8, color="#1f77b4", alpha=0.6, label="Rain %")
            ax2.set_ylim(0, 100)
            ax2.set_ylabel("Rain %")
            ax2.axhline(30, color="#4fa3ff", alpha=0.3, linewidth=1)
            ax2.axhline(60, color="#4fa3ff", alpha=0.3, linewidth=1)

            # The state ribbon (bottom strip):
            # each hour is a colored block (sunny/cloudy/rain/snow/thunder)
            states = today_hours["weather_state"].fillna("cloudy").tolist()
            colors = {
                'sunny': '#f0c419',
                'cloudy': '#9ca3af',
                'rain': '#1f77b4',
                'thunderstorm': '#9467bd',
                'snowstorm': '#17becf',
            }
            state_colors = [colors.get(s, '#9ca3af') for s in states]
            # Build a 1xN image of RGB values
            state_row = np.array([list(mcolors.to_rgb(c)) for c in state_colors]).reshape(1, -1, 3)
            self.t1_ax_state.imshow(state_row, aspect='auto', extent=[0, 24, 0, 1])
            self.t1_ax_state.set_yticks([])
            self.t1_ax_state.set_xlim(0, 24)
            self.t1_ax_state.set_xticks(range(0, 24, 3))
            self.t1_ax_state.set_xlabel("Hour (state ribbon)")
            # Add a small legend-like note
            self._add_note(
                self.t1_ax_main,
                "Orange line = temperature (°C)\nDashed = 3h average\nBlue bars = rain chance (%)\nBottom colors = weather state",
                loc='upper left'
            )
        else:
            self.t1_ax_main.text(0.5, 0.5, "No data for today", transform=self.t1_ax_main.transAxes, ha='center', va='center')
            self.t1_ax_state.axis('off')

        self.t1_left_fig.suptitle(f"{city} — Today")
        self.t1_left_canvas.draw_idle()

        # Right figure: Next 7 days — min/max ranges, daily mean ± std, and average rain chance
        self.t1_right_ax.clear(); self.t1_right_ax_rain.clear()
        start_date = today + timedelta(days=1)
        end_date = start_date + timedelta(days=6)
        dd_city = self.df_daily[self.df_daily["city"] == city].copy()
        dd_city["date_local"] = pd.to_datetime(dd_city["date_local"]).dt.date
        dd_future = dd_city[(dd_city["date_local"] >= start_date) & (dd_city["date_local"] <= end_date)].copy()

        hh_city = self.df_hourly[self.df_hourly["city"] == city].copy()
        hh_city["date_local"] = pd.to_datetime(hh_city["datetime_local"]).dt.date

        if not dd_future.empty:
            dd_future.sort_values("date_local", inplace=True)
            x = np.arange(len(dd_future))
            labels = [pd.to_datetime(d).strftime('%a %d') for d in dd_future["date_local"]]
            mins = dd_future["temp_min_c"].astype(float).values
            maxs = dd_future["temp_max_c"].astype(float).values
            means = dd_future["temp_mean_c"].astype(float).values
            rains = dd_future["precip_prob_avg_pct"].astype(float).values

            # For uncertainty: compute per-day temperature standard deviation from the hourly temps
            stds = []
            for d in dd_future["date_local"]:
                temps_d = hh_city.loc[hh_city["date_local"] == d, "temp_c"].astype(float)
                stds.append(float(temps_d.std()) if not temps_d.empty else np.nan)
            stds = np.array(stds)

            # Range: vlines for min/max, mean point with errorbar
            self.t1_right_ax.vlines(x, mins, maxs, color="#ff7f0e", linewidth=3, alpha=0.8)
            self.t1_right_ax.errorbar(x, means, yerr=stds, fmt='o', color="#ffd27f", ecolor="#ffd27f", capsize=3, label='Mean ± std')
            self.t1_right_ax.set_ylabel("°C")
            self.t1_right_ax.set_xticks(x, labels=labels, rotation=0)
            self.t1_right_ax.grid(True, axis='y', alpha=0.2)
            self.t1_right_ax.set_title(f"{city} — Next 7 Days: Min/Max ranges, Mean ± std")

            # Rain chance on twin axis
            self.t1_right_ax_rain.bar(x, rains, width=0.6, alpha=0.35, color="#1f77b4", label='Avg Rain %')
            self.t1_right_ax_rain.set_ylim(0, 100)
            self.t1_right_ax_rain.set_ylabel("Rain %")
            self.t1_right_ax_rain.axhline(30, color="#4fa3ff", alpha=0.3, linewidth=1)
            self.t1_right_ax_rain.axhline(60, color="#4fa3ff", alpha=0.3, linewidth=1)

            # Legends
            self.t1_right_ax.legend(loc='upper right')
            # Add a small legend-like note
            self._add_note(
                self.t1_right_ax,
                "Thick lines = min–max °C\nDot + whiskers = mean ± std\nBlue bars = daily rain chance (%)",
                loc='upper left'
            )
        else:
            self.t1_right_ax.text(0.5, 0.5, "No future data", transform=self.t1_right_ax.transAxes, ha='center', va='center')
            self.t1_right_ax_rain.set_ylim(0, 100)

        self.t1_right_canvas.draw_idle()

    def open_compare_popup(self):
        # Side-by-side mini dashboard to compare two cities for the next 7 days
        city_a = self.selected_city.get()
        if not city_a:
            return
        top = tk.Toplevel(self)
        top.title("Compare Cities — Tab 1")
        top.geometry("1400x900")
        try:
            top.configure(bg=APP_BG, highlightthickness=0)
        except Exception:
            pass

        # Controls (dark bar)
        ctl = tk.Frame(top, bg=CTRL_BG, highlightthickness=0, bd=0)
        ctl.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        tk.Label(ctl, text=f"City A: {city_a}", font=("Segoe UI", 10, "bold"), bg=CTRL_BG, fg=CTRL_TEXT).pack(side=tk.LEFT)
        tk.Label(ctl, text="City B:", bg=CTRL_BG, fg=CTRL_TEXT).pack(side=tk.LEFT, padx=(16, 4))
        city_b_var = tk.StringVar(value=city_a)
        cb = ttk.Combobox(ctl, textvariable=city_b_var, values=self.cities, state="readonly", width=24, style='City.TCombobox')
        cb.pack(side=tk.LEFT)

        body = tk.Frame(top, bg=APP_BG, highlightthickness=0, bd=0)
        body.pack(expand=True, fill=tk.BOTH, padx=0, pady=8)

        left = tk.Frame(body, bg=APP_BG, highlightthickness=0, bd=0)
        right = tk.Frame(body, bg=APP_BG, highlightthickness=0, bd=0)
        left.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=0)
        right.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=0)

        # Left city tables
        a_today, a_next = self._tab1_city_widgets(left)
        # Right city tables (initialize with A; update when selection changes)
        b_today, b_next = self._tab1_city_widgets(right)

        def render():
            city_b = city_b_var.get()
            self._fill_tab1_city(city_a, a_today, a_next)
            self._fill_tab1_city(city_b, b_today, b_next)

        cb.bind("<<ComboboxSelected>>", lambda e: render())
        render()

    def _tab1_city_widgets(self, parent: tk.Widget):
        # Create a small chart identical to Tab 1's right side (for compare dialog)
        fig, ax = plt.subplots(1, 1, figsize=(7, 4), constrained_layout=True)
        try:
            fig.patch.set_facecolor(APP_BG)
        except Exception:
            pass
        ax_rain = ax.twinx()
        try:
            ax.set_facecolor(APP_BG)
            ax_rain.set_facecolor(APP_BG)
        except Exception:
            pass
        canvas = FigureCanvasTkAgg(fig, master=parent)
        w = canvas.get_tk_widget()
        w.pack(fill=tk.BOTH, expand=True)
        try:
            w.configure(bg=APP_BG, highlightthickness=0, bd=0)
        except Exception:
            pass
        # store tuple so we can reuse drawing function
        return ((fig, ax, ax_rain, canvas), None)

    def _fill_tab1_city(self, city: str, today_widgets, next_tree):
        # Draw a city's next-7 view on the provided axes
        (fig, ax, ax_rain, canvas) = today_widgets
        ax.clear(); ax_rain.clear()
        today = self._today_date()
        dd_city = self.df_daily[self.df_daily["city"] == city].copy()
        dd_city["date_local"] = pd.to_datetime(dd_city["date_local"]).dt.date
        start_date = today + timedelta(days=1)
        end_date = start_date + timedelta(days=6)
        dd_future = dd_city[(dd_city["date_local"] >= start_date) & (dd_city["date_local"] <= end_date)].copy()
        hh_city = self.df_hourly[self.df_hourly["city"] == city].copy()
        hh_city["date_local"] = pd.to_datetime(hh_city["datetime_local"]).dt.date

        if not dd_future.empty:
            dd_future.sort_values("date_local", inplace=True)
            x = np.arange(len(dd_future))
            labels = [pd.to_datetime(d).strftime('%a %d') for d in dd_future["date_local"]]
            mins = dd_future["temp_min_c"].astype(float).values
            maxs = dd_future["temp_max_c"].astype(float).values
            means = dd_future["temp_mean_c"].astype(float).values
            rains = dd_future["precip_prob_avg_pct"].astype(float).values

            stds = []
            for d in dd_future["date_local"]:
                temps_d = hh_city.loc[hh_city["date_local"] == d, "temp_c"].astype(float)
                stds.append(float(temps_d.std()) if not temps_d.empty else np.nan)
            stds = np.array(stds)

            ax.vlines(x, mins, maxs, color="#ff7f0e", linewidth=3, alpha=0.8)
            ax.errorbar(x, means, yerr=stds, fmt='o', color="#ffd27f", ecolor="#ffd27f", capsize=3, label='Mean ± std')
            ax.set_ylabel("°C")
            ax.set_xticks(x, labels=labels, rotation=0)
            ax.grid(True, axis='y', alpha=0.2)
            ax.set_title(f"{city} — Next 7 Days")

            ax_rain.bar(x, rains, width=0.6, alpha=0.35, color="#1f77b4", label='Avg Rain %')
            ax_rain.set_ylim(0, 100)
            ax_rain.set_ylabel("Rain %")
            ax_rain.axhline(30, color="#4fa3ff", alpha=0.3, linewidth=1)
            ax_rain.axhline(60, color="#4fa3ff", alpha=0.3, linewidth=1)
            ax.legend(loc='upper right')
        else:
            ax.text(0.5, 0.5, "No future data", transform=ax.transAxes, ha='center', va='center')
            ax_rain.set_ylim(0, 100)

        canvas.draw_idle()

    # ---------- Tab 2 ----------
    # Tab 2: Quarters 2023/2024
    # Shows the distribution of daily mean temperatures for each quarter (Q1–Q4)
    # using violin plots, one grid for 2023 (left) and one for 2024 (right).
    # Each quarter panel is annotated with mean/min/max and the rainy‑day fraction.
    def _build_tab2(self):
        left = tk.Frame(self.tab2, bg=APP_BG, highlightthickness=0, bd=0)
        right = tk.Frame(self.tab2, bg=APP_BG, highlightthickness=0, bd=0)
        left.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(8, 4), pady=8)
        right.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(4, 8), pady=8)

        # Left figure (2023), Right figure (2024): each 2x2 violins/boxplots
        self.t2_fig_left = plt.Figure(figsize=(7.5, 6), constrained_layout=True)
        self.t2_fig_left.patch.set_facecolor(APP_BG)
        self.t2_ax_left = [self.t2_fig_left.add_subplot(2,2,i+1) for i in range(4)]
        for ax in self.t2_ax_left:
            try:
                ax.set_facecolor(APP_BG)
            except Exception:
                pass
        self.t2_canvas_left = FigureCanvasTkAgg(self.t2_fig_left, master=left)
        self.t2_canvas_left.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.t2_canvas_left.get_tk_widget().configure(bg=APP_BG, highlightthickness=0, bd=0)

        self.t2_fig_right = plt.Figure(figsize=(7.5, 6), constrained_layout=True)
        self.t2_fig_right.patch.set_facecolor(APP_BG)
        self.t2_ax_right = [self.t2_fig_right.add_subplot(2,2,i+1) for i in range(4)]
        for ax in self.t2_ax_right:
            try:
                ax.set_facecolor(APP_BG)
            except Exception:
                pass
        self.t2_canvas_right = FigureCanvasTkAgg(self.t2_fig_right, master=right)
        self.t2_canvas_right.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.t2_canvas_right.get_tk_widget().configure(bg=APP_BG, highlightthickness=0, bd=0)

    def _draw_t2_year(self, axes, df_daily_hist: pd.DataFrame, df_quarterly: pd.DataFrame, year: int, city: str):
        # Helper: draw one year (four quarters) into a 2x2 grid of axes
        # Violin plots show the full spread of daily means (not just min/max)
        quarters = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'}
        for idx, qn in enumerate([1,2,3,4]):
            ax = axes[idx]
            qlabel = quarters[qn]
            # Daily temps for this quarter
            mask = (df_daily_hist['city']==city) & (pd.to_datetime(df_daily_hist['date_local']).dt.year==year) & (pd.to_datetime(df_daily_hist['date_local']).dt.quarter==qn)
            temps = df_daily_hist.loc[mask, 'temp_mean_c'].astype(float).dropna().values
            ax.clear()
            if temps.size > 0:
                parts = ax.violinplot(temps, showmeans=True, showmedians=True)
                for pc in parts['bodies']:
                    pc.set_facecolor('#1f77b4'); pc.set_alpha(0.5)
                for k in ('cbars','cmins','cmaxes','cmeans','cmedians'):
                    if k in parts: parts[k].set_color('#ffd27f')
                # Add small summary labels in each panel
                mean_v = float(np.mean(temps))
                min_v = float(np.min(temps))
                max_v = float(np.max(temps))
                ax.set_title(f"{qlabel}")
                ax.set_xticks([])
                ax.set_ylabel("°C")
                ax.grid(True, alpha=0.2)
                ax.text(0.02, 0.95, f"mean {mean_v:.1f}\nmin {min_v:.1f}\nmax {max_v:.1f}", transform=ax.transAxes, va='top', ha='left', fontsize=9)
                # Rainy-day fraction (what fraction of days had any measurable rain)
                rowq = df_quarterly[(df_quarterly['year']==year) & (df_quarterly['quarter']==qlabel) & (df_quarterly['city']==city)]
                if not rowq.empty:
                    frac = float(rowq.iloc[0]['rainy_day_frac'])*100.0
                    ax.text(0.98, 0.95, f"rain {frac:.0f}%", transform=ax.transAxes, va='top', ha='right', fontsize=9, color='#4fa3ff')
            else:
                ax.text(0.5,0.5,"No data", ha='center', va='center')
                ax.set_title(f"{qlabel}")

    def _refresh_tab2(self):
        # Draw 2023 (left grid) and 2024 (right grid) for the selected city
        city = self.selected_city.get()
        if not city or self.h_quarterly.empty or self.h_daily.empty:
            return
        dfq_city = self.h_quarterly[self.h_quarterly["city"] == city]
        self._draw_t2_year(self.t2_ax_left, self.h_daily, dfq_city, 2023, city)
        self._draw_t2_year(self.t2_ax_right, self.h_daily, dfq_city, 2024, city)
        self.t2_fig_left.suptitle(f"2023 — {city}")
        self.t2_fig_right.suptitle(f"2024 — {city}")
        self.t2_canvas_left.draw_idle()
        self.t2_canvas_right.draw_idle()

    # ---------- Tab 3 ----------
    # Tab 3: Differences 23→24
    # Left column: (top) month-by-month change in mean temperature (2024 − 2023),
    #              (bottom) grouped bars comparing quarterly mean temperatures.
    # Right column: scatter of 2023 vs 2024 monthly means with a 45° reference line.
    def _build_tab3(self):
        self.t3_fig = plt.Figure(figsize=(12, 8), constrained_layout=True)
        self.t3_fig.patch.set_facecolor(APP_BG)
        gs = self.t3_fig.add_gridspec(2, 2)
        self.t3_ax1 = self.t3_fig.add_subplot(gs[0, 0])  # monthly delta
        self.t3_ax2 = self.t3_fig.add_subplot(gs[1, 0])  # quarterly grouped temps
        self.t3_ax3 = self.t3_fig.add_subplot(gs[:, 1])  # scatter 23 vs 24 or rainy frac grouped
        for ax in (self.t3_ax1, self.t3_ax2, self.t3_ax3):
            try:
                ax.set_facecolor(APP_BG)
            except Exception:
                pass
        self.t3_canvas = FigureCanvasTkAgg(self.t3_fig, master=self.tab3)
        self.t3_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.t3_canvas.get_tk_widget().configure(bg=APP_BG, highlightthickness=0, bd=0)

    def _refresh_tab3(self):
        # Recompute and redraw all three panels for the selected city
        city = self.selected_city.get()
        if not city or self.h_monthly.empty or self.h_quarterly.empty:
            return
        dfm = self.h_monthly[self.h_monthly["city"] == city].copy()
        # Panel 1: monthly temperature change (2024 − 2023)
        m23 = dfm[dfm["year"] == 2023].set_index("month")["temp_mean_c"]
        m24 = dfm[dfm["year"] == 2024].set_index("month")["temp_mean_c"]
        months = range(1, 13)
        delta = []
        for m in months:
            v23 = m23.get(m, np.nan)
            v24 = m24.get(m, np.nan)
            delta.append(v24 - v23 if (not np.isnan(v23) and not np.isnan(v24)) else np.nan)

        self.t3_ax1.clear()
        colors = ["#d62728" if (not np.isnan(d) and d >= 0) else "#1f77b4" for d in delta]
        self.t3_ax1.bar(list(months), delta, color=colors)
        self.t3_ax1.axhline(0, color="#444", linewidth=1)
        self.t3_ax1.set_title(f"Monthly Temperature Δ (2024 − 2023) — {city}")
        self.t3_ax1.set_xlabel("Month")
        self.t3_ax1.set_ylabel("Δ °C")
        self.t3_ax1.set_xticks(list(months))
        # Value labels on each bar for quick reading
        for i, d in enumerate(delta, start=1):
            if not np.isnan(d):
                self.t3_ax1.text(i, d + (0.1 if d>=0 else -0.1), f"{d:.1f}", ha='center', va='bottom' if d>=0 else 'top', fontsize=8)
        # Small explainer
        self._add_note(self.t3_ax1, "Bars = 2024 − 2023 monthly mean °C", loc='upper left')

        # Panel 2: quarterly grouped bars for mean temperature (2023 vs 2024)
        dfq = self.h_quarterly[self.h_quarterly["city"] == city].copy()
        q_order = ["Q1", "Q2", "Q3", "Q4"]
        q23 = dfq[dfq["year"] == 2023].set_index("quarter")
        q24 = dfq[dfq["year"] == 2024].set_index("quarter")
        self.t3_ax2.clear()
        idx = np.arange(len(q_order))
        width = 0.35
        mean23 = [q23.get("temp_mean_c").get(q, np.nan) for q in q_order]
        mean24 = [q24.get("temp_mean_c").get(q, np.nan) for q in q_order]
        self.t3_ax2.bar(idx - width/2, mean23, width, label="2023")
        self.t3_ax2.bar(idx + width/2, mean24, width, label="2024")
        self.t3_ax2.set_xticks(idx, q_order)
        self.t3_ax2.set_ylabel("Mean Temp (°C)")
        self.t3_ax2.set_title("Quarterly Mean Temperatures: 2023 vs 2024")
        self.t3_ax2.legend()
        self._add_note(self.t3_ax2, "Paired bars compare 2023 vs 2024", loc='upper left')

        # Panel 3 (right): scatter of 2023 vs 2024 monthly means
        # Each point is a month; points above the dashed line mean 2024 was warmer.
        self.t3_ax3.clear()
        xvals = [m23.get(m, np.nan) for m in months]
        yvals = [m24.get(m, np.nan) for m in months]
        self.t3_ax3.scatter(xvals, yvals, c=["#ffd27f" if (not np.isnan(x) and not np.isnan(y)) else '#888' for x,y in zip(xvals, yvals)])
        # 45-degree reference
        allv = [v for v in xvals + yvals if not np.isnan(v)]
        if allv:
            vmin, vmax = min(allv), max(allv)
            pad = (vmax - vmin) * 0.1 if vmax>vmin else 1
            self.t3_ax3.plot([vmin-pad, vmax+pad], [vmin-pad, vmax+pad], color="#444", linestyle='--')
            self.t3_ax3.set_xlim(vmin-pad, vmax+pad)
            self.t3_ax3.set_ylim(vmin-pad, vmax+pad)
        self.t3_ax3.set_xlabel("2023 Monthly Mean °C")
        self.t3_ax3.set_ylabel("2024 Monthly Mean °C")
        self.t3_ax3.set_title("Monthly Means: 2024 vs 2023")
        self._add_note(self.t3_ax3, "Above line = 2024 warmer; below = cooler", loc='upper left')

        self.t3_canvas.draw_idle()

    # ---------- Tab 4 ----------
    # Tab 4: Projection (baseline + anomaly)
    # Not a forecast: we take the average of 2023/2024 as a baseline and adjust
    # remaining months by this year's year‑to‑date anomaly. We show temperature
    # and rainy‑day fraction as separate dumbbell charts.
    def _build_tab4(self):
        self.t4_fig, (self.t4_ax1, self.t4_ax2) = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
        self.t4_fig.patch.set_facecolor(APP_BG)
        for ax in (self.t4_ax1, self.t4_ax2):
            try:
                ax.set_facecolor(APP_BG)
            except Exception:
                pass
        self.t4_canvas = FigureCanvasTkAgg(self.t4_fig, master=self.tab4)
        self.t4_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.t4_canvas.get_tk_widget().configure(bg=APP_BG, highlightthickness=0, bd=0)

    def _refresh_tab4(self):
        # Compute a simple “projection” for remaining months based on a baseline + anomaly idea
        city = self.selected_city.get()
        if not city or self.h_monthly.empty:
            return
        dfm = self.h_monthly[self.h_monthly["city"] == city].copy()
        today = pd.Timestamp.today().date()
        current_year = today.year

        # Baseline per month = average of 2023 and 2024
        m23 = dfm[dfm["year"] == 2023].set_index("month")
        m24 = dfm[dfm["year"] == 2024].set_index("month")
        # Remaining months in this calendar year (after the current month)
        months_rem = [m for m in range(1, 13) if m >= today.month + 1]
        if not months_rem:
            months_rem = []

        # Year‑to‑date anomaly: how this year's average so far compares to the 23/24 baseline so far
        my = dfm[dfm["year"] == current_year].set_index("month")
        ytd_months = [m for m in range(1, today.month + 1)]
        if ytd_months:
            avg_ytd = my.loc[ytd_months, "temp_mean_c"].mean(skipna=True) if set(ytd_months).issubset(my.index) else np.nan
            avg_baseline_ytd = pd.concat([m23.loc[ytd_months, "temp_mean_c"], m24.loc[ytd_months, "temp_mean_c"]], axis=1).mean(axis=1).mean(skipna=True) if (set(ytd_months).issubset(m23.index) and set(ytd_months).issubset(m24.index)) else np.nan
            anomaly = (avg_ytd - avg_baseline_ytd) if (not np.isnan(avg_ytd) and not np.isnan(avg_baseline_ytd)) else 0.0
        else:
            anomaly = 0.0

        # Build temperature projection for the remaining months: baseline + the same anomaly
        proj_months = months_rem
        baseline_temp = []
        proj_temp = []
        for m in proj_months:
            base = np.nanmean([m23.get("temp_mean_c").get(m, np.nan), m24.get("temp_mean_c").get(m, np.nan)])
            baseline_temp.append(base)
            proj_temp.append(base + anomaly if not np.isnan(base) else np.nan)

        self.t4_ax1.clear()
        idx = np.arange(len(proj_months))
        # Compute simple uncertainty from 23/24 std (may be small)
        m23_vals = [m23.get("temp_mean_c").get(m, np.nan) for m in proj_months]
        m24_vals = [m24.get("temp_mean_c").get(m, np.nan) for m in proj_months]
        temp_std = [np.nanstd([a, b], ddof=1) if (not np.isnan(a) and not np.isnan(b)) else 0.0 for a, b in zip(m23_vals, m24_vals)]
        # Dumbbell plot per month
        for i, (base, proj, err) in enumerate(zip(baseline_temp, proj_temp, temp_std)):
            if np.isnan(base) or np.isnan(proj):
                continue
            # connecting line
            self.t4_ax1.plot([i, i], [base, proj], color="#888", alpha=0.7, linewidth=2)
            # baseline: hollow marker + optional errorbar
            self.t4_ax1.errorbar(i, base, yerr=err if err else None, fmt='none', ecolor="#ffd27f", elinewidth=1, capsize=3, alpha=0.7)
            self.t4_ax1.scatter(i, base, s=60, facecolors='none', edgecolors="#ffd27f", label="Baseline (23–24 avg)" if i == 0 else None, zorder=3)
            # projected: filled marker
            self.t4_ax1.scatter(i, proj, s=60, color="#ff7f0e", label=f"Projected {current_year}" if i == 0 else None, zorder=3)
        self.t4_ax1.set_xticks(idx, [str(m) for m in proj_months])
        self.t4_ax1.set_ylabel("Mean Temp (°C)")
        self.t4_ax1.set_title(f"Projected Remaining Months — {city}")
        self.t4_ax1.grid(True, axis='y', alpha=0.2)
        self.t4_ax1.legend(loc='upper right')
        self._add_note(self.t4_ax1, "Dumbbell: circle = baseline, filled = projected\nLine shows change per month", loc='upper left')

        # Rainy-day fraction projection (multiplicative anomaly):
        # If this year's rainy-day fraction so far is higher than the 23/24 baseline,
        # we scale the remaining months by that ratio.
        # Compute the YTD ratio first.
        if ytd_months and set(ytd_months).issubset(my.index) and set(ytd_months).issubset(m23.index) and set(ytd_months).issubset(m24.index):
            my_ytd_rain = my.loc[ytd_months, "rainy_day_frac"].mean(skipna=True) if "rainy_day_frac" in my.columns else np.nan
            base_ytd_rain = pd.concat([m23.loc[ytd_months, "rainy_day_frac"], m24.loc[ytd_months, "rainy_day_frac"]], axis=1).mean(axis=1).mean(skipna=True)
            if (base_ytd_rain is not None and not np.isnan(base_ytd_rain) and my_ytd_rain is not None and not np.isnan(my_ytd_rain) and base_ytd_rain != 0):
                ratio = my_ytd_rain / base_ytd_rain
            else:
                ratio = 1.0
        else:
            ratio = 1.0

        baseline_rain = []
        proj_rain = []
        rain_std = []
        for m in proj_months:
            r23 = m23.get("rainy_day_frac").get(m, np.nan)
            r24 = m24.get("rainy_day_frac").get(m, np.nan)
            base_r = np.nanmean([r23, r24])
            rs = np.nanstd([r23, r24], ddof=1) if (not np.isnan(r23) and not np.isnan(r24)) else 0.0
            baseline_rain.append(base_r)
            rain_std.append(rs)
            proj_rain.append(base_r * ratio if (base_r is not None and not np.isnan(base_r)) else np.nan)

        self.t4_ax2.clear()
        # Dumbbell plot for rainy-day fraction
        for i, (base, proj, err) in enumerate(zip(baseline_rain, proj_rain, rain_std)):
            if base is None or np.isnan(base) or proj is None or np.isnan(proj):
                continue
            base_pct = base * 100.0
            proj_pct = proj * 100.0
            err_pct = (err * 100.0) if err else None
            self.t4_ax2.plot([i, i], [base_pct, proj_pct], color="#888", alpha=0.7, linewidth=2)
            if err_pct:
                self.t4_ax2.errorbar(i, base_pct, yerr=err_pct, fmt='none', ecolor="#ffd27f", elinewidth=1, capsize=3, alpha=0.7)
            self.t4_ax2.scatter(i, base_pct, s=60, facecolors='none', edgecolors="#ffd27f", label="Baseline (23–24 avg)" if i == 0 else None, zorder=3)
            self.t4_ax2.scatter(i, proj_pct, s=60, color="#1f77b4", label=f"Projected {current_year}" if i == 0 else None, zorder=3)
        self.t4_ax2.set_xticks(idx, [str(m) for m in proj_months])
        self.t4_ax2.set_ylabel("Rainy-Day Fraction (%)")
        self.t4_ax2.set_title("Projected Raininess (fraction of rainy days)")
        # Not a forecast badge
        self.t4_ax2.text(0.99, 0.01, "Not a forecast — baseline+anomaly", transform=self.t4_ax2.transAxes, ha='right', va='bottom', fontsize=8, color='#bbb')
        self.t4_ax2.grid(True, axis='y', alpha=0.2)
        self.t4_ax2.legend(loc='upper right')
        self._add_note(self.t4_ax2, "Dumbbell: circle = baseline, filled = projected\nLine shows change per month", loc='upper left')
        self.t4_ax2.legend()

        self.t4_canvas.draw_idle()

    # ---------- Refresh all tabs ----------
    def refresh_all(self):
        # Re-render all tabs (called at startup and when the city selection changes)
        self._refresh_tab1()
        self._refresh_tab2()
        self._refresh_tab3()
        self._refresh_tab4()


def main():
    # Launch the app window and start the Tkinter event loop
    app = StormyApp()
    app.mainloop()


if __name__ == "__main__":
    main()
