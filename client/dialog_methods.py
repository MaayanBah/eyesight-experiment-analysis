import tkinter as tk
from tkinter import filedialog, messagebox


def display_message(message: str) -> None:
    """
    :param message: A message to show the user
    :return: None
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    messagebox.showinfo("Message", message)


def get_directory_path(title) -> str:
    """
    :param title: The title of the directory dialog.
    :return: The path for the directory the user chose.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    directory_path: str = filedialog.askdirectory(
        title=title,
        initialdir="/path/to/default/directory")

    if directory_path:
        print("You chose the directory:", directory_path)
    else:
        raise NotADirectoryError("No directory selected")
    return directory_path


def show_yes_no_dialog(question: str) -> bool:
    """
    :param question: A question for the user.
    :return: A boolean value, True is the user chose 'yes' and 'false' otherwise.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    return messagebox.askyesno("Question", question)
