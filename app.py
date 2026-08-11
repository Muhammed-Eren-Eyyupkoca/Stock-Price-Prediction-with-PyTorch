"""Desktop GUI for the BIST-30 stock price prediction pipeline.

Home screen shows the app name and two info buttons (purpose / how-to-use)
that expand an inline text panel below them. Picking a symbol from the navy
sidebar switches to its analysis screen: its opening price for the day is
fetched immediately, and an "Analiz Et" button runs fetch -> preprocess ->
train -> evaluate in the background, then shows the RMSE/MAE/MAPE metrics
and the loss/prediction plots.

Usage:
    python app.py
"""

import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import matplotlib

matplotlib.use("Agg")  # headless backend: plotting happens off the main (Tk) thread

from PIL import Image, ImageTk  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parent
SYMBOLS_FILE = ROOT_DIR / "scripts" / "symbols" / "bist30.txt"
IMAGE_DISPLAY_WIDTH = 560

sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from evaluate import RESULTS_DIR, plot_predictions_vs_actual, run_evaluation  # noqa: E402
from fetch_data import fetch_and_save, get_opening_price  # noqa: E402
from preprocess import preprocess_symbol  # noqa: E402
from train import train_model  # noqa: E402

# --- Renk paleti -------------------------------------------------------
NAVY = "#132743"
NAVY_DARK = "#0B1B2E"
NAVY_HOVER = "#1E3A5F"
NAVY_ACTIVE = "#081422"
NAVY_SELECTED = "#2F6FED"
MAIN_BG = "#F5F7FB"
TEXT_MUTED = "#5B6B85"

ABOUT_TEXT = (
    "Bu proje, PyTorch kullanarak geçmiş hisse senedi verilerinden gelecekteki "
    "fiyat hareketlerini tahmin etmeyi amaçlayan bir zaman serisi tahmin "
    "uygulamasıdır.\n\n"
    "LSTM ve GRU mimarilerinin performansını karşılaştırarak hisse senedi fiyat "
    "tahmininde hangi modelin daha başarılı olduğunu ortaya koymayı hedefler.\n\n"
    "Veriler Yahoo Finance (yfinance) üzerinden alınır; fiyat ve teknik "
    "göstergeler (SMA-20, RSI-14, MACD) özellik olarak kullanılır."
)

HOWTO_TEXT = (
    "1. Sol menüden bir BIST-30 hissesi seçin.\n\n"
    "2. Seçilen hissenin güne başlangıç fiyatı ekranda görünür.\n\n"
    "3. 'Analiz Et' butonuna basın. Uygulama sırasıyla veriyi indirir, "
    "ön işler, bir LSTM modeli eğitir (25 epoch) ve test setinde değerlendirir. "
    "Bu işlem yaklaşık 1-2 dakika sürebilir; arayüz bu sırada kilitlenmez.\n\n"
    "4. Analiz tamamlanınca RMSE / MAE / MAPE metrikleri ile loss eğrisi ve "
    "gerçek/tahmin fiyat grafiği gösterilir.\n\n"
    "Not: Her yeni analiz, bir önceki analizin kaydedilmiş verisinin/modelinin "
    "üzerine yazar."
)


def load_symbols() -> list[str]:
    """Reads BIST-30 ticker symbols from scripts/symbols/bist30.txt (skips
    blank lines and '#' comments)."""
    symbols = []
    for line in SYMBOLS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            symbols.append(line)
    return symbols


def run_pipeline(symbol: str, log: "queue.Queue") -> dict:
    """Runs fetch -> preprocess -> train -> evaluate for `symbol`, posting
    progress messages to `log` as it goes.

    Returns:
        The run_evaluation() result dict (dates/actuals/preds/rmse/mae/mape).

    Raises:
        RuntimeError: If fetching data for `symbol` fails.
        FileNotFoundError: If a downstream step's required input is missing.
    """
    log.put(f"[1/4] '{symbol}' için veri indiriliyor...")
    if not fetch_and_save(symbol, years=5):
        raise RuntimeError(f"'{symbol}' için veri indirilemedi.")

    log.put("[2/4] Ön işleme (teknik göstergeler + train/val/test bölme)...")
    counts = preprocess_symbol(symbol)
    log.put(f"      train={counts['train']}, val={counts['val']}, test={counts['test']} satır.")

    log.put("[3/4] LSTM modeli eğitiliyor (25 epoch)...")
    train_result = train_model(log=log.put)
    log.put(f"      En iyi validation loss: {train_result['best_val_loss']:.6f}")

    log.put("[4/4] Test setinde değerlendiriliyor...")
    eval_result = run_evaluation()
    plot_predictions_vs_actual(
        eval_result["dates"], eval_result["actuals"], eval_result["preds"],
        RESULTS_DIR / "predictions_vs_actual.png",
    )

    log.put("Analiz tamamlandı.")
    return eval_result


class App(tk.Tk):
    """Ana pencere: karşılama ekranı + lacivert BIST-30 sembol menüsü +
    seçilen hisse için güne başlangıç fiyatı / analiz ekranı."""

    def __init__(self) -> None:
        super().__init__()
        self.title("HisseAnaliz")
        self.geometry("1280x860")
        self.minsize(1020, 720)
        self.configure(bg=MAIN_BG)

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        self.log_queue: "queue.Queue" = queue.Queue()
        self.selected_symbol = ""
        self.symbol_buttons: dict[str, tk.Button] = {}
        self._is_running = False
        self.status_text = tk.StringVar(value="")
        self._loss_photo = None
        self._pred_photo = None

        self._build_layout()
        self.after(100, self._poll_log_queue)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        container = tk.Frame(self, bg=MAIN_BG)
        container.pack(fill="both", expand=True)

        self._build_sidebar(container)

        self.pages = tk.Frame(container, bg=MAIN_BG)
        self.pages.pack(side="left", fill="both", expand=True)
        self.pages.grid_rowconfigure(0, weight=1)
        self.pages.grid_columnconfigure(0, weight=1)

        self.home_page = tk.Frame(self.pages, bg=MAIN_BG)
        self.home_page.grid(row=0, column=0, sticky="nsew")
        self._build_home_page()

        self.analysis_page = tk.Frame(self.pages, bg=MAIN_BG)
        self.analysis_page.grid(row=0, column=0, sticky="nsew")
        self._build_analysis_page()

        self.home_page.tkraise()

    def _build_sidebar(self, parent: tk.Widget) -> None:
        sidebar = tk.Frame(parent, width=230, bg=NAVY_DARK)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Label(
            sidebar, text="BIST-30 Hisseleri", font=("Segoe UI", 12, "bold"),
            bg=NAVY_DARK, fg="white",
        ).pack(anchor="w", padx=14, pady=(16, 10))

        list_area = tk.Frame(sidebar, bg=NAVY_DARK)
        list_area.pack(fill="both", expand=True, padx=(10, 0), pady=(0, 10))

        canvas = tk.Canvas(list_area, bg=NAVY_DARK, highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_area, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        buttons_frame = tk.Frame(canvas, bg=NAVY_DARK)
        window_id = canvas.create_window((0, 0), window=buttons_frame, anchor="nw")
        buttons_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(window_id, width=e.width))

        def _bind_scroll(_event: object) -> None:
            canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))

        def _unbind_scroll(_event: object) -> None:
            canvas.unbind_all("<MouseWheel>")

        canvas.bind("<Enter>", _bind_scroll)
        canvas.bind("<Leave>", _unbind_scroll)

        for symbol in load_symbols():
            btn = tk.Button(
                buttons_frame,
                text=symbol,
                bg=NAVY,
                fg="white",
                activebackground=NAVY_ACTIVE,
                activeforeground="white",
                relief="flat",
                bd=0,
                anchor="w",
                font=("Segoe UI", 10),
                cursor="hand2",
                padx=14,
                pady=9,
                command=lambda s=symbol: self._on_symbol_selected(s),
            )
            btn.pack(fill="x", padx=6, pady=2)
            btn.bind("<Enter>", lambda _e, b=btn, s=symbol: self._on_button_hover(b, s, True))
            btn.bind("<Leave>", lambda _e, b=btn, s=symbol: self._on_button_hover(b, s, False))
            self.symbol_buttons[symbol] = btn

    def _on_button_hover(self, button: tk.Button, symbol: str, entering: bool) -> None:
        if symbol == self.selected_symbol:
            return
        button.config(bg=NAVY_HOVER if entering else NAVY)

    def _build_home_page(self) -> None:
        self._info_open_key: str | None = None
        self._info_buttons: dict[str, tk.Button] = {}
        self._info_content = {"about": ABOUT_TEXT, "howto": HOWTO_TEXT}

        center = tk.Frame(self.home_page, bg=MAIN_BG)
        center.pack(expand=True)

        tk.Label(
            center, text="HisseAnaliz", font=("Segoe UI", 34, "bold"), bg=MAIN_BG, fg=NAVY,
        ).pack()
        tk.Label(
            center,
            text="BIST-30 hisseleri için LSTM tabanlı fiyat tahmin uygulaması",
            font=("Segoe UI", 11),
            bg=MAIN_BG,
            fg=TEXT_MUTED,
        ).pack(pady=(6, 28))

        button_row = tk.Frame(center, bg=MAIN_BG)
        button_row.pack()

        self._info_buttons["about"] = tk.Button(
            button_row, text="Projenin Amacı", command=lambda: self._toggle_info_panel("about"),
            bg=NAVY, fg="white", activebackground=NAVY_ACTIVE, activeforeground="white",
            relief="flat", bd=0, font=("Segoe UI", 10, "bold"), cursor="hand2", padx=18, pady=11,
        )
        self._info_buttons["about"].pack(side="left", padx=8)

        self._info_buttons["howto"] = tk.Button(
            button_row, text="Nasıl Kullanılır", command=lambda: self._toggle_info_panel("howto"),
            bg=NAVY, fg="white", activebackground=NAVY_ACTIVE, activeforeground="white",
            relief="flat", bd=0, font=("Segoe UI", 10, "bold"), cursor="hand2", padx=18, pady=11,
        )
        self._info_buttons["howto"].pack(side="left", padx=8)

        self.info_panel = tk.Frame(
            center, bg="white", highlightthickness=1, highlightbackground="#DCE3EE"
        )
        self.info_label = tk.Label(
            self.info_panel, text="", justify="left", wraplength=560,
            bg="white", fg="#2B3B55", font=("Segoe UI", 10), padx=22, pady=18,
        )
        self.info_label.pack(fill="both", expand=True)

    def _toggle_info_panel(self, key: str) -> None:
        if self._info_open_key == key:
            self.info_panel.pack_forget()
            self._info_buttons[key].config(bg=NAVY)
            self._info_open_key = None
            return

        if self._info_open_key is not None:
            self._info_buttons[self._info_open_key].config(bg=NAVY)

        self.info_label.config(text=self._info_content[key])
        self.info_panel.pack(fill="x", pady=(22, 0))
        self._info_buttons[key].config(bg=NAVY_SELECTED)
        self._info_open_key = key

    def _build_analysis_page(self) -> None:
        page = self.analysis_page
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(7, weight=1)
        pad = {"padx": 28}

        self.symbol_title_label = tk.Label(
            page, text="", font=("Segoe UI", 22, "bold"), bg=MAIN_BG, fg=NAVY
        )
        self.symbol_title_label.grid(row=0, column=0, sticky="w", pady=(24, 0), **pad)

        self.price_label = tk.Label(
            page, text="", font=("Segoe UI", 15), bg=MAIN_BG, fg="#2B3B55"
        )
        self.price_label.grid(row=1, column=0, sticky="w", pady=(4, 16), **pad)

        self.analyze_button = tk.Button(
            page, text="Analiz Et", command=self._on_analyze_click,
            bg=NAVY, fg="white", activebackground=NAVY_ACTIVE, activeforeground="white",
            relief="flat", bd=0, font=("Segoe UI", 11, "bold"), cursor="hand2",
            padx=20, pady=10, state="disabled",
        )
        self.analyze_button.grid(row=2, column=0, sticky="w", pady=(0, 10), **pad)

        ttk.Label(
            page, textvariable=self.status_text, background=MAIN_BG, foreground=TEXT_MUTED,
            font=("Segoe UI", 9),
        ).grid(row=3, column=0, sticky="w", pady=(0, 4), **pad)

        self.progress = ttk.Progressbar(page, mode="indeterminate")
        self.progress.grid(row=4, column=0, sticky="ew", pady=(0, 8), **pad)
        self.progress.grid_remove()

        self.log_frame = ttk.LabelFrame(page, text="Günlük")
        self.log_frame.grid(row=5, column=0, sticky="ew", pady=8, **pad)
        self.log_text = tk.Text(self.log_frame, height=7, state="disabled", wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=4, pady=4)
        self.log_frame.grid_remove()

        self.metrics_frame = ttk.LabelFrame(page, text="Sonuç Metrikleri")
        self.metrics_frame.grid(row=6, column=0, sticky="ew", pady=(0, 8), **pad)
        self.metric_labels = {}
        for name in ("RMSE", "MAE", "MAPE"):
            label = ttk.Label(self.metrics_frame, text=f"{name}: -", font=("Segoe UI", 10, "bold"))
            label.pack(side="left", padx=16, pady=8)
            self.metric_labels[name] = label
        self.metrics_frame.grid_remove()

        self.images_frame = tk.Frame(page, bg=MAIN_BG)
        self.images_frame.grid(row=7, column=0, sticky="nsew", **pad)
        self.loss_image_label = tk.Label(self.images_frame, bg=MAIN_BG)
        self.loss_image_label.pack(side="left", fill="both", expand=True, padx=(0, 4))
        self.pred_image_label = tk.Label(self.images_frame, bg=MAIN_BG)
        self.pred_image_label.pack(side="left", fill="both", expand=True, padx=(4, 0))

    # ------------------------------------------------------------------
    # Sembol seçimi + güne başlangıç fiyatı
    # ------------------------------------------------------------------
    def _on_symbol_selected(self, symbol: str) -> None:
        if self._is_running:
            return

        if self.selected_symbol and self.selected_symbol in self.symbol_buttons:
            self.symbol_buttons[self.selected_symbol].config(bg=NAVY)
        self.selected_symbol = symbol
        self.symbol_buttons[symbol].config(bg=NAVY_SELECTED)

        self.analysis_page.tkraise()
        self.symbol_title_label.config(text=symbol)
        self.price_label.config(text="Güne başlangıç fiyatı alınıyor...")
        self.analyze_button.config(state="normal")
        self.status_text.set("")
        self._clear_log()
        self.progress.grid_remove()
        self.log_frame.grid_remove()
        self.metrics_frame.grid_remove()
        self.loss_image_label.config(image="")
        self.pred_image_label.config(image="")

        threading.Thread(target=self._fetch_price_thread, args=(symbol,), daemon=True).start()

    def _fetch_price_thread(self, symbol: str) -> None:
        try:
            price = get_opening_price(symbol)
            self.log_queue.put(("__price__", symbol, price))
        except Exception as exc:
            self.log_queue.put(("__price_error__", symbol, str(exc)))

    def _on_price_fetched(self, symbol: str, price: float) -> None:
        if symbol != self.selected_symbol:
            return
        self.price_label.config(text=f"Güne Başlangıç Fiyatı: {price:.2f} TL")

    def _on_price_error(self, symbol: str, _message: str) -> None:
        if symbol != self.selected_symbol:
            return
        self.price_label.config(text="Güne başlangıç fiyatı alınamadı.")

    # ------------------------------------------------------------------
    # Analiz akışı
    # ------------------------------------------------------------------
    def _on_analyze_click(self) -> None:
        symbol = self.selected_symbol
        if not symbol or self._is_running:
            return

        self._is_running = True
        self.analyze_button.config(state="disabled")
        for button in self.symbol_buttons.values():
            button.config(state="disabled")
        self._clear_log()
        self.log_frame.grid()
        self.metrics_frame.grid_remove()
        self.progress.grid()
        self.progress.start(12)
        self.status_text.set(f"'{symbol}' analiz ediliyor...")

        threading.Thread(target=self._run_pipeline_thread, args=(symbol,), daemon=True).start()

    def _run_pipeline_thread(self, symbol: str) -> None:
        try:
            result = run_pipeline(symbol, self.log_queue)
            self.log_queue.put(("__done__", symbol, result))
        except Exception as exc:  # surfaced in the UI instead of crashing the thread
            self.log_queue.put(("__error__", symbol, str(exc)))

    def _poll_log_queue(self) -> None:
        try:
            while True:
                item = self.log_queue.get_nowait()
                if isinstance(item, tuple) and item[0] == "__done__":
                    self._on_pipeline_done(item[1], item[2])
                elif isinstance(item, tuple) and item[0] == "__error__":
                    self._on_pipeline_error(item[1], item[2])
                elif isinstance(item, tuple) and item[0] == "__price__":
                    self._on_price_fetched(item[1], item[2])
                elif isinstance(item, tuple) and item[0] == "__price_error__":
                    self._on_price_error(item[1], item[2])
                else:
                    self._append_log(str(item))
        except queue.Empty:
            pass
        self.after(100, self._poll_log_queue)

    def _append_log(self, message: str) -> None:
        self.log_text.config(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _clear_log(self) -> None:
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

    def _finish_run(self) -> None:
        self._is_running = False
        self.progress.stop()
        self.progress.grid_remove()
        self.analyze_button.config(state="normal")
        for symbol, button in self.symbol_buttons.items():
            button.config(state="normal", bg=NAVY_SELECTED if symbol == self.selected_symbol else NAVY)

    def _on_pipeline_done(self, symbol: str, result: dict) -> None:
        self._finish_run()
        self.status_text.set(f"'{symbol}' için analiz tamamlandı.")

        self.metric_labels["RMSE"].config(text=f"RMSE: {result['rmse']:.2f}")
        self.metric_labels["MAE"].config(text=f"MAE: {result['mae']:.2f}")
        self.metric_labels["MAPE"].config(text=f"MAPE: %{result['mape']:.2f}")
        self.metrics_frame.grid()

        self._loss_photo = self._load_image(RESULTS_DIR / "loss_curve.png")
        self.loss_image_label.config(image=self._loss_photo)
        self._pred_photo = self._load_image(RESULTS_DIR / "predictions_vs_actual.png")
        self.pred_image_label.config(image=self._pred_photo)

    def _on_pipeline_error(self, symbol: str, message: str) -> None:
        self._finish_run()
        self.status_text.set(f"Hata: '{symbol}' analiz edilemedi.")
        self._append_log(f"HATA: {message}")

    @staticmethod
    def _load_image(path: Path) -> ImageTk.PhotoImage:
        image = Image.open(path)
        ratio = IMAGE_DISPLAY_WIDTH / image.width
        image = image.resize(
            (IMAGE_DISPLAY_WIDTH, int(image.height * ratio)), Image.Resampling.LANCZOS
        )
        return ImageTk.PhotoImage(image)


if __name__ == "__main__":
    App().mainloop()
