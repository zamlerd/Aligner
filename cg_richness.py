import csv
import scipy.special
import numpy

# ignore warnings caused by zero probability states
numpy.seterr(divide = "ignore")

sequence_alphabet = "ACGT"
hidden_state_alphabet = "SRP" # start, CG rich, CG poor
neginf = float("-inf")

sequence = "GGCACTGAA"

n = len(sequence) # the length of the sequence and index of the last position
m = 3  # the number of states

# this is a numeric encoding of the sequence, where
# A = 0
# C = 1
# G = 2
# T = 3
# the first character is also offset by 1, for pseudo-1-based-addressing
numeric_sequence = numpy.zeros(n + 1, dtype = numpy.uint8)
for i in range(n):
	numeric_sequence[i + 1] = sequence_alphabet.index(sequence[i])

# emission probabilities, row x column = numeric nucleotide code x state k
e_matrix = numpy.array([
	[0.0, 0.13, 0.37],
	[0.0, 0.37, 0.13],
	[0.0, 0.37, 0.13],
	[0.0, 0.13, 0.37]
])

# transition probabilities, row x column = state k x state k'
t_matrix = numpy.array([
	[0.00, 0.50, 0.50],
	[0.00, 0.63, 0.37],
	[0.00, 0.37, 0.63]
])

log_e_matrix = numpy.log(e_matrix)
log_t_matrix = numpy.log(t_matrix)

# all calculations will be in log space

v_matrix = numpy.zeros((n + 1, m)) # Viterbi log probabilities
r_matrix = numpy.zeros((n + 1, m), dtype = numpy.uint8) # Viterbi pointers
b_matrix = numpy.zeros((n + 1, m)) # Backward log probabilities

# initialize matrix probabilities
v_matrix.fill(neginf)
b_matrix.fill(neginf)
v_matrix[0,0] = 0.0
b_matrix[n] = 0.0

# temp matrices used to store the log probabilities for all
# state transitions to or from each state k at position i
viterbi_temp = numpy.zeros(m)

for i in range(1, n + 1):
	for k in range(m): # state at i
		for j in range(m): # state at i - 1
			e = log_e_matrix[numeric_sequence[i], k] # emission log probability
			t = log_t_matrix[j, k] # transition log probability
			v = v_matrix[i - 1, j] # recursive log probability

			viterbi_temp[j] = e + t + v

		v_matrix[i, k] = numpy.max(viterbi_temp)
		r_matrix[i, k] = numpy.argmax(viterbi_temp)

backward_temp = numpy.zeros(m)

for i in reversed(range(n)):
	for k in range(m): # state at i
		for j in range(m): # state at i + 1
			e = log_e_matrix[numeric_sequence[i + 1], j]
			t = log_t_matrix[k, j]
			b = b_matrix[i + 1, j]

			backward_temp[j] = e + t + b

		b_matrix[i, k] = scipy.special.logsumexp(backward_temp)

# initialize the maximum a posteriori hidden state path using the state with
# the highest joint probability at the last position
map_path = numpy.zeros(n + 1, dtype = numpy.uint8)
map_state = numpy.argmax(v_matrix[n])
map_path[n] = map_state

# then follow the pointers backwards
for i in reversed(range(n)):
	map_path[i] = r_matrix[i + 1, map_state]
	map_state = map_path[i]

# and convert the state k indices into a sensical string
map_path_string = ""
for i in range(1, n + 1):
	map_state = map_path[i]
	map_path_string += hidden_state_alphabet[map_state]

# apply Bayes rule to get the posterior probability
marginal_likelihood = b_matrix[0, 0]
map_joint_probability = numpy.max(v_matrix[n])
map_posterior_probability = numpy.exp(map_joint_probability - marginal_likelihood)

print("Maximum a posteriori path = %s, posterior probability = %.2f%%" % (
	map_path_string,
	map_posterior_probability * 100
	))
