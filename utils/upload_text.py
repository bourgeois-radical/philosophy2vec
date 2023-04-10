import pathlib


def get_text(path: pathlib.WindowsPath) -> str:
    """Reads any .txt file with the text the model should be trained on.

    Parameters
    ----------
    path: pathlib.WindowsPath

    Returns
    -------
    text_string : str
        The whole text as a string.
    """

    with open(path, 'r', encoding='utf8') as text:
        # reading the file to string
        text_string = text.read()

        # getting rid of words such as 'hearing-and-interrogating-of-itself' etc. for future tokenization
    text_string = text_string.replace("-", " ")

    return text_string
