import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog


def display_message(message: str) -> None:
    """
    :param message: A message to show the user
    :return: None
    """
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("Message", message)


def get_directory_path(title) -> str:
    """
    :param title: The title of the directory dialog.
    :return: The path for the directory the user chose.
    """
    root = tk.Tk()
    root.withdraw()
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
    root.withdraw()
    return messagebox.askyesno("Question", question)


def get_user_text_through_dialog(title: str, prompt: str) -> str:
    """
    :param prompt: A prompt for the user.
    :param title: A title for the dialog box.
    :return: The string input from the user.
    """
    root = tk.Tk()
    root.withdraw()
    user_input = simpledialog.askstring(title, prompt)
    if user_input is None:
        raise ValueError("No input provided")
    return user_input
