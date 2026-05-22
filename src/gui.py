import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from tkinter import messagebox
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from simulator import simulate_euler, exact_mean, exact_variance

SIDEBAR_BG   = "#1b1b2f"
SIDEBAR_FG   = "#e2e2e2"
SECTION_FG   = "#8a8aa0"
ENTRY_BG     = "#27274a"
ENTRY_FG     = "#ffffff"
SEPARATOR    = "#3b3b5c"
ACCENT       = "#5c6bc0"
ACCENT_HOVER = "#7986cb"
MAIN_BG      = "#f4f4f8"

class SDESimulatorApp:
    """Ventana principal del simulador de EDE autónoma."""

    _COEFF_PARAMS = [
        ("a", "Drift lineal  (a)", "1.0"),
        ("b", "Drift constante  (b)", "0.0"),
        ("c", "Difusión lineal  (c)", "0.5"),
        ("d", "Difusión constante  (d)", "0.0"),
    ]

    _SIM_PARAMS = [
        ("X0", "Condición inicial  (X₀)", "1.0"),
        ("T",  "Tiempo final  (T)", "1.0"),
        ("dt", "Paso de tiempo  (Δt)", "0.01"),
        ("M",  "Trayectorias  (M)", "20"),
    ]

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Simulador de EDE Autónoma — Euler-Maruyama")
        self.root.geometry("1300x780")
        self.root.minsize(960, 600)
        self.root.configure(bg=MAIN_BG)

        self.entries: dict[str, tk.Entry] = {}

        self._build_sidebar()
        self._build_plot_area()

    def _build_sidebar(self):
        sb = tk.Frame(self.root, bg=SIDEBAR_BG, width=310)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        sb.pack_propagate(False)

        tk.Label(
            sb,
            text="Parámetros",
            font=("Segoe UI", 16, "bold"),
            bg=SIDEBAR_BG,
            fg=SIDEBAR_FG,
        ).pack(pady=(20, 4), padx=20, anchor="w")

        tk.Label(
            sb,
            text="dXₜ = (aXₜ + b)dt + (cXₜ + d)dBₜ",
            font=("Consolas", 10),
            bg=SIDEBAR_BG,
            fg=ACCENT_HOVER,
        ).pack(padx=20, anchor="w", pady=(0, 12))

        self._separator(sb)

        self._section_label(sb, "COEFICIENTES DE LA EDE")

        for key, label, default in self._COEFF_PARAMS:
            self._param_row(sb, key, label, default)

        self._separator(sb)

        self._section_label(sb, "PARÁMETROS DE SIMULACIÓN")

        for key, label, default in self._SIM_PARAMS:
            self._param_row(sb, key, label, default)

        btn_frame = tk.Frame(sb, bg=SIDEBAR_BG)
        btn_frame.pack(fill=tk.X, padx=20, pady=25)

        self.sim_btn = tk.Button(
            btn_frame,
            text="SIMULAR",
            font=("Segoe UI", 13, "bold"),
            bg=ACCENT,
            fg="white",
            activebackground=ACCENT_HOVER,
            activeforeground="white",
            bd=0,
            pady=10,
            cursor="hand2",
            command=self.run_simulation,
        )
        self.sim_btn.pack(fill=tk.X)

        # NUEVO BOTÓN LIMPIAR
        self.clear_btn = tk.Button(
            btn_frame,
            text="LIMPIAR",
            font=("Segoe UI", 11, "bold"),
            bg="#424242",
            fg="white",
            activebackground="#616161",
            activeforeground="white",
            bd=0,
            pady=8,
            cursor="hand2",
            command=self.reset_app,
        )
        self.clear_btn.pack(fill=tk.X, pady=(10, 0))

        self.status_var = tk.StringVar(value="Listo para simular")

        tk.Label(
            sb,
            textvariable=self.status_var,
            font=("Segoe UI", 9),
            bg=SIDEBAR_BG,
            fg=SECTION_FG,
        ).pack(side=tk.BOTTOM, padx=20, pady=10, anchor="w")

    def _separator(self, parent):
        tk.Frame(parent, height=1, bg=SEPARATOR).pack(
            fill=tk.X,
            padx=15,
            pady=8,
        )

    def _section_label(self, parent, text):
        tk.Label(
            parent,
            text=text,
            font=("Segoe UI", 9, "bold"),
            bg=SIDEBAR_BG,
            fg=SECTION_FG,
        ).pack(padx=20, anchor="w", pady=(6, 4))

    def _param_row(self, parent, key, label, default):
        frame = tk.Frame(parent, bg=SIDEBAR_BG)
        frame.pack(fill=tk.X, padx=20, pady=2)

        tk.Label(
            frame,
            text=label,
            font=("Segoe UI", 10),
            bg=SIDEBAR_BG,
            fg=SIDEBAR_FG,
            anchor="w",
        ).pack(anchor="w")

        entry = tk.Entry(
            frame,
            font=("Consolas", 11),
            bg=ENTRY_BG,
            fg=ENTRY_FG,
            insertbackground=ENTRY_FG,
            bd=0,
            relief="flat",
            highlightthickness=1,
            highlightbackground=SEPARATOR,
            highlightcolor=ACCENT,
        )

        entry.pack(fill=tk.X, pady=(2, 0), ipady=4)
        entry.insert(0, default)

        entry.bind("<Return>", lambda _: self.run_simulation())

        self.entries[key] = entry

    def _build_plot_area(self):
        plot_frame = tk.Frame(self.root, bg=MAIN_BG)
        plot_frame.pack(
            side=tk.RIGHT,
            fill=tk.BOTH,
            expand=True,
            padx=8,
            pady=8,
        )

        self.fig = Figure(figsize=(9, 6), facecolor="#fafafa", dpi=100)

        self.ax = self.fig.add_subplot(111)

        self._empty_plot()

        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()

        # NUEVA TOOLBAR DE NAVEGACIÓN
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()

        # Widget del gráfico
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _empty_plot(self):
        ax = self.ax

        ax.set_facecolor("#fafafa")

        ax.set_title(
            "Simulación de EDE Autónoma",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )

        ax.set_xlabel("Tiempo  t", fontsize=12)
        ax.set_ylabel("X(t)", fontsize=12)

        ax.grid(True, alpha=0.2, linestyle="--")

        ax.text(
            0.5,
            0.5,
            "Presione  SIMULAR  para comenzar",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            color="#b0b0b0",
        )

    # NUEVO MÉTODO RESET
    def reset_app(self):
        """Restaura parámetros por defecto y limpia la gráfica."""

        for key, _, default in self._COEFF_PARAMS:
            self.entries[key].delete(0, tk.END)
            self.entries[key].insert(0, default)

        for key, _, default in self._SIM_PARAMS:
            self.entries[key].delete(0, tk.END)
            self.entries[key].insert(0, default)

        self.ax.clear()

        self._empty_plot()

        self.canvas.draw()

        self.status_var.set("Parámetros restaurados")

    def _read_params(self):
        """Lee y valida todos los campos. Retorna dict o None si hay error."""

        try:
            v = {}

            for key in ("a", "b", "c", "d", "X0", "T", "dt"):
                raw = self.entries[key].get().strip()
                v[key] = float(raw)

            v["M"] = int(self.entries["M"].get().strip())

            if v["T"] <= 0:
                raise ValueError("El tiempo final T debe ser positivo.")

            if v["dt"] <= 0:
                raise ValueError("El paso de tiempo Δt debe ser positivo.")

            if v["dt"] > v["T"]:
                raise ValueError("Δt debe ser menor o igual que T.")

            if v["M"] <= 0:
                raise ValueError("M debe ser un entero positivo.")

            return v

        except ValueError as exc:
            messagebox.showerror("Error en parámetros", str(exc))
            return None

    def run_simulation(self):
        vals = self._read_params()

        if vals is None:
            return

        a  = vals["a"]
        b  = vals["b"]
        c  = vals["c"]
        d  = vals["d"]
        X0 = vals["X0"]
        T  = vals["T"]
        dt = vals["dt"]
        M  = vals["M"]

        self.status_var.set("Simulando …")

        self.root.update_idletasks()

        try:
            t_grid, X_paths = simulate_euler(a, b, c, d, X0, T, dt, M)

            m = exact_mean(a, b, X0, t_grid)

            var = exact_variance(a, b, c, d, X0, t_grid)

            std = np.sqrt(var)

        except Exception as exc:
            messagebox.showerror("Error de simulación", str(exc))
            self.status_var.set("Error")
            return

        ax = self.ax

        ax.clear()

        ax.set_facecolor("#fafafa")

        alpha_traj = max(0.08, 0.6 / np.sqrt(max(M, 1)))

        for i in range(M):

            label_traj = "Trayectorias (Euler)" if i == 0 else None

            ax.plot(
                t_grid,
                X_paths[i],
                color="#42A5F5",
                alpha=alpha_traj,
                linewidth=0.8,
                label=label_traj,
                zorder=1,
            )

        ax.fill_between(
            t_grid,
            m - std,
            m + std,
            color="#66BB6A",
            alpha=0.15,
            zorder=2,
        )

        ax.plot(
            t_grid,
            m + std,
            color="#43A047",
            linewidth=2,
            linestyle="--",
            label="E[Xₜ] ± σ(Xₜ)",
            zorder=3,
        )

        ax.plot(
            t_grid,
            m - std,
            color="#43A047",
            linewidth=2,
            linestyle="--",
            zorder=3,
        )

        ax.plot(
            t_grid,
            m,
            color="#EF5350",
            linewidth=2.5,
            label="Media exacta  E[Xₜ]",
            zorder=4,
        )

        n_steps = len(t_grid) - 1

        title = (
            f"dX = ({a}·X + {b})dt + ({c}·X + {d})dBₜ"
            f"   │   M = {M},  Δt = {dt},  n = {n_steps}"
        )

        ax.set_title(title, fontsize=11, fontweight="bold", pad=12)

        ax.set_xlabel("Tiempo  t", fontsize=12)

        ax.set_ylabel("X(t)", fontsize=12)

        ax.legend(loc="best", fontsize=10, framealpha=0.9)

        ax.grid(True, alpha=0.2, linestyle="--")

        self.fig.tight_layout()

        self.canvas.draw()

        self.status_var.set(
            f"✓  {M} trayectorias  ·  {n_steps} pasos  ·  Δt = {dt}"
        )

def main():
    root = tk.Tk()
    SDESimulatorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()