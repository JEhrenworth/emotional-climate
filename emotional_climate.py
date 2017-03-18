import re
import os
import sys
import numpy as np
from operator import itemgetter

clean_data_objects = []
DELIMITER = "---"


class Data(object):
	"""
	Provides a simple infrastructure for storing feeling related data.  No error-checking is done, and it is assumed that the data coming in has already been cleaned.

	Attributes:
		feeling (str): The feeling attached to the dataset.
		data (str): The dataset.
		weights (list): The dataset mapped to floating point numbers.
		average_weight (float): The average of the weights.  Everytime weights is updated, average_weight is recomputed.
	"""

	def __init__(self, feeling, data):
		self.feeling = feeling
		self.data = data
		self.weights = []
		self.average_weight = 0

	def update_weights(self, new_weights):
		"""
		Changes the weight attribute to new_weight.  The average weight is recomputed everytime this method is called.

		Args:
			new_weights (list): A list of floating point or integer objects.
		"""
		self.weights = new_weights
		self.update_average_weight()

	def update_average_weight(self):
		"""
		Calculates a new average weight.
		"""
		self.average_weight = np.mean(self.weights)


def main():
	raw_data = get_data_from_file("EmotionalClimateData.dat", DELIMITER, 5, 15)
	cleaned_data = clean_data(raw_data)

	for data in cleaned_data:
		populate_dataset(data[0], data[1])

	for data_object in clean_data_objects:
		data_object.update_weights(find_weights(data_object.data))

	order_weights(clean_data_objects)


def get_data_from_file(file_to_read_from, delimiter, lower_index=0, upper_index=None):
	"""
	Specialty function to grab raw data from a file, split by a given delimeter between a certain index.  This is not a wholy safe function, and does not cleanly raise an error message for every input.

	Args:
		file_to_read_from (str): The name of the file to open.
		delimiter (str): The character to use to seperate the data.
		lower_index (int): After the data is split, the lower index of the data to return.  Defaults to zero.  I.e. beggining of the data.
		upper_index (int): After the data is split, the upper index of the data to return.  Defaults to None.  If no upper_index is supplied, then the data from after lower will be returned.

	Returns:
		A string of raw data.

	Raises:
		FileNotFoundError: Raises an exception if the file to read data from can not be found.

		>>> get_data_from_file("not-real-file.dat", DELIMITER)
		Traceback (most recent call last):
			...
			...
			...
		FileNotFoundError: not-real-file.dat could not be found.
	"""
	if not os.path.isfile(file_to_read_from):
		raise FileNotFoundError("{file_to_read_from} could not be found.".format(file_to_read_from=file_to_read_from))

	with open(file_to_read_from, 'r') as data_file:
		data = data_file.read().split(delimiter)

		if upper_index is not None:
			return data[lower_index:upper_index]
		else:
			return data[lower_index:len(data)]


def clean_data(raw_data):
	"""
	Attempts to clean given list of raw data and return a tuple of formatted data.  If expected errors are raised while trying to clean an element of data, the data is skipped and the next element is cleaned.

	Args:
		raw_data (list): The data to attempt to clean.

	Returns:
		An array filled with tuples of the form (feeling, data_attached_to_feeling)

		>>> clean_data(["\nHow do you feel when you're at school? [Supported]\nOften\nAlways\nOften\nOften\nSometimes\nNever"])
		[('Supported', 'Often Always Often Often Sometimes Never')]

		>>> clean_data(["\nHow do you feel when you're at school? [Bored]\nNever\nSometimes\nOften", "\nHow do you feel when you're at school? [Stressed]\nSometimes\nOften\nSometimes\nOften"])
		[('Bored', 'Never Sometimes Often'), ('Stressed', 'Sometimes Often Sometimes Often')]

		If the data can't be read, and an expected error message is raised, it will be passed for the next set of data.

		>>> clean_data(["Timestamp\n3/12/2017 18:50:08\n", "\nHow do you feel when you're at school? [Tired]\nSometimes\nAlways\nSometimes\n"])
		An error was raised when trying to extract a feeling from the input data.  This set of data will be skipped.  The error raised was: Subgroup must be less than the number of words in brackets in the dataset.  There were 0 word(s) matched, and the subgroup argument supplied was 0.
		[('Tired', 'Sometimes Always Sometimes')]

	Raises:
		TypeError: Raises an exception if a raw_data isn't a list, or if the elements of raw_data aren't all strings.

		>>> clean_data(2.718)
		Traceback (most recent call last):
			...
			...
			...
		TypeError: Argument given was of type <class 'float'>, but must be of type list.

		>>> clean_data(["2.718", 1.20205])
		Traceback (most recent call last):
			...
			...
			...
		TypeError: Argument given was a list, but not every element of that list was a string.
	"""

	if not isinstance(raw_data, list):
		raise TypeError("Argument given was of type {raw_data_type}, but must be of type list.".format(raw_data_type=type(raw_data)))

	if not all(isinstance(data, str) for data in raw_data):
		raise TypeError("Argument given was a list, but not every element of that list was a string.")

	cleaned_data_tuple = []

	for data in raw_data:
		try:
			clean_feeling = extract_feeling(data)
		except (TypeError, IndexError) as error:
			print("An error was raised when trying to extract a feeling from the input data.  This set of data will be skipped.  The error raised was: {error}".format(error=error.with_traceback(sys.exc_info()[2])))
		else:
			try:
				cleaned_data = strip_data(data)
			except TypeError as error:
				print("An error was raised when trying to extract data from the input data.  This set of data will be skipped.  The error raised was: {error}".format(error=error.with_traceback(sys.exc_info()[2])))
			else:
				cleaned_data_tuple.append((clean_feeling, cleaned_data))

	if cleaned_data_tuple:
		return cleaned_data_tuple


def extract_feeling(data, subgroup=0):
	"""
	Finds a word between brackets from some data set.  If no word is present then an empty string is returned.  In the case that multiple words are between brackets, this by default returns only the first word between brackets.  The nth word (zero indexed, of course) can be obtained by supplying the optional argument subgroup.

	Args:
		data (str): A string of data to find the word from.
		subgroup (int): Which word to return, if multiple words are found.  Default argument is zero.  I.e. return first word found.

	Returns:
		A string representing the word between brackets in the dataset, or an empty string if no brackets are present.

		>>> extract_feeling("[Hurt] Never Always Sometimes")
		"Hurt"

		>>> extract_feeling("Nothing to return here")
		""

		>>> extract_feeling("[6] [90]")
		"6"

		>>> extract_feeling("[6] [90]", subgroup=1)
		"90"

	Raises:
		TypeError: Raises an exception if data is not a string.

		>>> extract_feeling(42)
		Traceback (most recent call last):
			...
			...
			...
		TypeError: 42 is of type <class 'int'>, not of type string.  This function only takes in strings as the data argument.

		IndexError: Raises an exception if subgroup is past the range of the list returned from re.findall(...) or if subgroup is less than zero.

		>>> extract_feeling("[945] [9450]", subgroup=2)
		Traceback (most recent call last):
			...
			...
			...
		IndexError: Subgroup must be less than the number of words in brackets in the dataset.  There were 2 word(s) matched, and the subgroup argument supplied was 2.

		>>> extract_feeling("[93555]", subgroup=-2)
		Traceback (most recent call last):
			...
			...
			...
		IndexError: Subgroup must be a positive number or zero.  A subgroup argument of -2 was supplied.

	"""

	if not isinstance(data, str):
		raise TypeError("{data} is of type {type}, not of type string.  This function only takes in a string as the data argument.".format(data=repr(data), type=type(data)))

	if subgroup < 0:
		raise IndexError("Subgroup must be a positive number or zero.  A subgroup argument of {subgroup} was supplied.".format(subgroup=subgroup))

	match = re.findall(r"\[(\w+)\]", data)
	number_of_matches = len(match)

	if subgroup >= number_of_matches:
		raise IndexError("Subgroup must be less than the number of words in brackets in the dataset.  There were {words_matched} word(s) matched, and the subgroup argument supplied was {subgroup}.".format(words_matched=number_of_matches, subgroup=subgroup))

	if match is not None:
		return match[subgroup]
	else:
		return ""


def strip_data(raw_data):
	"""
	Replaces all newlines with spaces and removes everything before the first closing bracket in the string.

	Args:
		raw_data (str): A string of data to clean.

	Returns:
		A new string with the cleaned data.

		>>> strip_data("\nHow do you feel when you're at school? [Tired]\nSometimes\nAlways\nSometimes")
		"Sometimes Always Sometimes"

		>>> strip_data("\nHow do you feel when you're at school? [Bored] [Loquacious] \nNever\nSometimes\nOften\nOften\nSometimes\nNever\nSometimes\nSometimes\nSometimes\nAlways\nOften\nNever")
		"[Loquacious]  Never Sometimes Often Often Sometimes Never Sometimes Sometimes Sometimes Always Often Never"

	Raises:
		TypeError: Raises an exception if the argument provided isn't a string.

		>>> strip_data(-1)
		Traceback (most recent call last):
			...
			...
			...
		TypeError: -1 is of type <class 'int'>, not of type string.  This function only takes in a string as an argument.
	"""

	if not isinstance(raw_data, str):
		raise TypeError("{raw_data} is of type {type}, not of type string.  This function only takes in a string as an argument.".format(raw_data=repr(raw_data), type=type(raw_data)))

	cleaned = raw_data.replace("\n", " ")
	return cleaned[cleaned.find(']') + 1:].strip()


def populate_dataset(feeling, feeling_data):
	"""
	Appends a Data object to cleaned_data_objects.

	Args:
		feeling (str): A string to use for the feeling perameter of the Data object.
		feeling_data (str): A string to use for the data parameter of the Data object.

	Raises:
		TypeError: Raises an exception if either one of the arguments supplied isn't a string.

		>>> populate_dataset("Logistical", 4.6692)
		Traceback (most recent call last):
			...
			...
			...
		TypeError: Feeling argument supplied was of type <class 'str'>, and feeling_data argument supplied was of type <class 'float'>.  They must both be of type string.
	"""

	if not isinstance(feeling, str) or not isinstance(feeling_data, str):
		raise TypeError("Feeling argument supplied was of type {feeling_type}, and feeling_data argument supplied was of type {data_type}.  They must both be of type string.".format(feeling_type=type(feeling), data_type=type(feeling_data)))

	clean_data_objects.append(Data(feeling, feeling_data))


def find_weights(data):
	"""
	Returns a list of number of range 0-1, mapping every "Never" to 0, every "Sometimes" to 1/3, every "Often" to 2/3, and every "Always" to 1.  Words with no mapping value will be mapped to None.

	Args:
		data (str): A string to map from.

	Returns:
		A new list with values mapped.  If there are occurances of words which have no mapping, they are replaced with None.

		>>> find_weights("Never Always Often")
		[0, 1, 0.6666666666666666]

		>>> find_weights("Always Never The Velvet Undergound Syzygy Mandelbrot Never")
		[1, 0, None, None, None, None, None, 0]

	Raises:
		TypeError: Raises an exception if something other than a string is supplied as the argument.

		>>> find_weights(1.618)
		Traceback (most recent call last):
			...
			...
			...
		TypeError: Data argument supplied was of type <class 'float'>, it must be of type string.
	"""

	if not isinstance(data, str):
		raise TypeError("Data argument supplied was of type {data_type}, it must be of type string.".format(data_type=type(data)))

	weight_for_time_interval = {"Never": 0, "Sometimes": 1 / 3, "Often": 2 / 3, "Always": 1}

	weights = []
	for time_interval in data.split(" "):
		if time_interval in weight_for_time_interval:
			weights.append(weight_for_time_interval[time_interval])
		else:
			weights.append(None)

	return weights


def order_weights(data_objects):
	weights = []

	for data_object in data_objects:
		weights.append((data_object.feeling, data_object.average_weight))

	weights.sort(key=itemgetter(1), reverse=True)

	for element in weights:
		print(element[0], element[1])


if __name__ == "__main__":
	main()
