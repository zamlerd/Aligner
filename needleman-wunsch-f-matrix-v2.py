# coding=utf8

import numpy

# The alphabet defines the order of amino acids used for the score matrix
# so A is the first character, R is the second etc.
alphabet = "ARNDCQEGHILKMFPSTWYV"

# The BLOSUM50 matrix in third-bit units, or 3log2(odds ratio)
# We store it as a 2D array using the numpy package
blosum50 = numpy.array([
	[ 5,-2,-1,-2,-1,-1,-1, 0,-2,-1,-2,-1,-1,-3,-1, 1, 0,-3,-2, 0],
	[-2, 7,-1,-2,-4, 1, 0,-3, 0,-4,-3, 3,-2,-3,-3,-1,-1,-3,-1,-3],
	[-1,-1, 7, 2,-2, 0, 0, 0, 1,-3,-4, 0,-2,-4,-2, 1, 0,-4,-2,-3],
	[-2,-2, 2, 8,-4, 0, 2,-1,-1,-4,-4,-1,-4,-5,-1, 0,-1,-5,-3,-4],
	[-1,-4,-2,-4,13,-3,-3,-3,-3,-2,-2,-3,-2,-2,-4,-1,-1,-5,-3,-1],
	[-1, 1, 0, 0,-3, 7, 2,-2, 1,-3,-2, 2, 0,-4,-1, 0,-1,-1,-1,-3],
	[-1, 0, 0, 2,-3, 2, 6,-3, 0,-4,-3, 1,-2,-3,-1,-1,-1,-3,-2,-3],
	[ 0,-3, 0,-1,-3,-2,-3, 8,-2,-4,-4,-2,-3,-4,-2, 0,-2,-3,-3,-4],
	[-2, 0, 1,-1,-3, 1, 0,-2,10,-4,-3, 0,-1,-1,-2,-1,-2,-3, 2,-4],
	[-1,-4,-3,-4,-2,-3,-4,-4,-4, 5, 2,-3, 2, 0,-3,-3,-1,-3,-1, 4],
	[-2,-3,-4,-4,-2,-2,-3,-4,-3, 2, 5,-3, 3, 1,-4,-3,-1,-2,-1, 1],
	[-1, 3, 0,-1,-3, 2, 1,-2, 0,-3,-3, 6,-2,-4,-1, 0,-1,-3,-2,-3],
	[-1,-2,-2,-4,-2, 0,-2,-3,-1, 2, 3,-2, 7, 0,-3,-2,-1,-1, 0, 1],
	[-3,-3,-4,-5,-2,-4,-3,-4,-1, 0, 1,-4, 0, 8,-4,-3,-2, 1, 4,-1],
	[-1,-3,-2,-1,-4,-1,-1,-2,-2,-3,-4,-1,-3,-4,10,-1,-1,-4,-3,-3],
	[ 1,-1, 1, 0,-1, 0,-1, 0,-1,-3,-3, 0,-2,-3,-1, 5, 2,-4,-2,-2],
	[ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 2, 5,-3,-2, 0],
	[-3,-3,-4,-5,-5,-1,-3,-3,-3,-3,-2,-3,-1, 1,-4,-4,-3,15, 2,-3],
	[-2,-1,-2,-3,-3,-1,-2,-3, 2,-1,-1,-2, 0, 4,-3,-2,-2, 2, 8,-1],
	[ 0,-3,-3,-4,-1,-3,-3,-4,-4, 4, 1,-3, 1,-1,-3,-2, 0,-3,-1, 5],
])

# Specify a linear GAP penalty of -8
gap_penalty = -8

# Our X and Y sequences to align
x = "MAMRLLKTHL"
y = "MKNITCYL"

# Set up our F-matrix using a two dimensional score_matrix to record the
# intermediate values and a three dimensional pointer_matrix to record the
# arrows. The first and second axes of the score_matrix and pointer_matrix
# correspond to the X and Y sequences respectively. The third axis of the
# pointer_matrix records the presence (1) or absence (0) of arrows in the
# diagonal (align both), left (align X with gap), and up (align Y with gap) at
# indices 0, 1 and 2 respectively. Because with have the first row and first
# column filled in with gap penalties, the row and column indices
# corresponding to the X and Y residues will begin at 1.
score_matrix = numpy.zeros((len(x) + 1, len(y) + 1), dtype = int)
pointer_matrix = numpy.zeros((len(x) + 1, len(y) + 1, 3), dtype = int)

# Fill in first row and column with the linear gap penalties
for xi, xaa in enumerate(x):
	score_matrix[xi + 1, 0] = gap_penalty * (xi + 1)
	pointer_matrix[xi + 1, 0, 1] = 1

for yi, yaa in enumerate(y):
	score_matrix[0, yi + 1] = gap_penalty * (yi + 1)
	pointer_matrix[0, yi + 1, 2] = 1

# Fill in middle values starting, from left to right in the top row, then from
# left to right in the second row and so on. The yi and xi indices begin at 0.
for yi, yaa in enumerate(y):
	for xi, xaa in enumerate(x):
		# Get the index (position) of the X and Y amino acids in our alphabet
		xaai = alphabet.index(xaa)
		yaai = alphabet.index(yaa)

		# These are the indices to the X and Y positions in the F-matrix,
		# which begin at 1 because of the the first row and first column of
		# gap penalties.
		fxi = xi + 1
		fyi = yi + 1

		# Look up the score for the X and Y pair of residues in our BLOSUM50 matrix
		residue_score = blosum50[xaai, yaai]

		# Calculate the possible scores for aligning both residues...
		no_gap_score = score_matrix[fxi - 1, fyi - 1] + residue_score # (fxi - 1, fyi - 1) is the diagonally up and left cell
		# ...and for aligning the X residue with a gap...
		x_gap_score = score_matrix[fxi - 1, fyi] + gap_penalty # (fxi - 1, fyi) is the cell to the left of the current cell
		# ...and for aligning the Y residue with a gap.
		y_gap_score = score_matrix[fxi, fyi - 1] + gap_penalty # (fxi, fyi - 1) is the cell above the current cell

		# Identify the maximum score value from among the three options...
		max_score = max(no_gap_score, x_gap_score, y_gap_score)
		# ...and set the score of this F-matrix cell to that score.
		score_matrix[fxi, fyi] = max_score

		# Add arrows for any direction where choosing that direction
		# will result in the maximum score for this F-matrix cell
		if no_gap_score == max_score:
			pointer_matrix[fxi, fyi, 0] = 1
		if x_gap_score == max_score:
			pointer_matrix[fxi, fyi, 1] = 1
		if y_gap_score == max_score:
			pointer_matrix[fxi, fyi, 2] = 1

# Print the Needleman-Wunsch F-matrix with scores and pointers to the command line
x_length = len(x) + 1
y_length = len(y) + 1
print_matrix = numpy.zeros([y_length * 2, x_length * 2], dtype = "U3")
print_matrix.fill("   ")

for xi in range(x_length):
	for yi in range(y_length):
		print_matrix[yi * 2 + 1][xi * 2 + 1] = "%3d" % score_matrix[xi][yi]

		if pointer_matrix[xi][yi][0]: # diagonal arrow
			print_matrix[yi * 2][xi * 2] = u" ↖ "
		if pointer_matrix[xi][yi][1]: # left arrow
			print_matrix[yi * 2 + 1][xi * 2] = u" ← "
		if pointer_matrix[xi][yi][2]: # up arrow
			print_matrix[yi * 2][xi * 2 + 1] = u" ↑ "

for row in print_matrix[1:]:
	print("".join(row[1:]))
