import tkinter as tk

from buzzsuite_app import VideoAnalyzerApp


def main():
    root = tk.Tk()
    app = VideoAnalyzerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()