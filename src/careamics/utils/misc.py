from typing import Optional


def int_to_3chars(number: Optional[int]) -> str:
    """Convert an integer to a fixed-length string of 3 characters.

    Parameters
    ----------
    number : int
        The input integer.

    Returns
    -------
    str
        A 3-character string derived from the integer.
    """
    if number is None:
        return "???"
    
    if number < 0:
        raise ValueError("Only non-negative integers are supported.")

    # Define the alphabet for the hash (base-62: 0-9, A-Z, a-z)
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    base = len(alphabet)

    # Convert number to the given base
    encoded = []
    while number > 0:
        encoded.append(alphabet[number % base])
        number //= base

    # Reverse the encoded result to get the correct order
    encoded.reverse()

    # Ensure the string is exactly 3 characters long
    hash_string = ''.join(encoded).zfill(3)

    # If the string is longer than 3 characters, truncate it
    return hash_string[-3:]

def get_sample_id(file_id: Optional[int], sample_id: Optional[int]) -> str:
    """Get the sample ID from the file and sample IDs.

    Parameters
    ----------
    file_id : Optional[int]
        The ID of the file, if available.
    sample_id : Optional[int]
        The ID of the sample, if available.

    Returns
    -------
    str
        The sample ID.
    """
    return f"{int_to_3chars(file_id)}-{int_to_3chars(sample_id)}"