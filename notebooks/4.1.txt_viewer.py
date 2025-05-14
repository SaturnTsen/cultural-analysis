import os
import tkinter as tk
from tkinter import scrolledtext


class TxtViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("TXT 快速查看器")
        self.text_area = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, font=("Courier", 12))
        self.text_area.pack(expand=True, fill='both')

        self.txt_files = [
            os.path.join(dirpath, filename)
            for dirpath, _, filenames in os.walk('C:\\Users\\ming\\OneDrive\\本科\\法语毕业论文\\project\\data\\cleaned')
            for filename in filenames if filename.endswith('.txt')
        ]

        self.txt_files.sort()
        self.index = 0

        self.root.bind("<Left>", self.prev_file)
        self.root.bind("<Right>", self.next_file)

        if self.txt_files:
            self.load_file()

    def load_file(self):
        self.text_area.delete('1.0', tk.END)
        filename = self.txt_files[self.index]
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            self.text_area.insert(tk.END, content)
        self.root.title(f"{filename} ({self.index + 1}/{len(self.txt_files)})")

    def prev_file(self, event=None):
        if self.index > 0:
            self.index -= 1
            self.load_file()

    def next_file(self, event=None):
        if self.index < len(self.txt_files) - 1:
            self.index += 1
            self.load_file()


if __name__ == "__main__":
    root = tk.Tk()
    viewer = TxtViewer(root)
    root.mainloop()
