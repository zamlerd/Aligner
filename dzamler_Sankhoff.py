#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ete3
import numpy
import os.path
import sys

neginf = float("-inf")
inf = float("inf")
cost_matrix = numpy.array([
	[ 0, -1, -1, -1],
	[-1, 0, -1, -1],
	[-1, -1, 0, -1],
	[-1, -1, -1, 0],
])

# used by read_fasta to turn a sequence string into a vector of integers based
# on the supplied alphabet
def vectorize_sequence(sequence, alphabet):
	sequence_length = len(sequence)

	sequence_vector = numpy.zeros(sequence_length, dtype = numpy.uint8)

	for i, char in enumerate(sequence):
		sequence_vector[i] = alphabet.index(char)
	return sequence_vector


# In[2]:


# this is a function that reads in a multiple sequence alignment stored in
# FASTA format, and turns it into a matrix
def read_fasta(fasta_path, alphabet):
	label_order = []
	sequence_matrix = numpy.zeros(0, dtype = numpy.uint8)

	fasta_file = open(fasta_path)

	l = fasta_file.readline()
	while l != "":
		l_strip = l.rstrip() # strip out newline characters

		if l[0] == ">":
			label = l_strip[1:]
			label_order.append(label)
		else:
			sequence_vector = vectorize_sequence(l_strip, alphabet)
			sequence_matrix = numpy.concatenate((sequence_matrix, sequence_vector))

		l = fasta_file.readline()

	fasta_file.close()

	n_sequences = len(label_order)
	sequence_length = len(sequence_matrix) // n_sequences
	sequence_matrix = sequence_matrix.reshape(n_sequences, sequence_length)
	return label_order, sequence_matrix


# In[3]:


# this is a function that reads in a phylogenetic tree stored in newick
# format, and turns it into an ete3 tree object
def read_newick(newick_path):
	newick_file = open(newick_path)
	newick = newick_file.read().strip()
	newick_file.close()

	tree = ete3.Tree(newick)
	return tree


# In[4]:


def has_sequence(node):
    try:
        this_seq = node.sequence
        return True
    except:
        return False


# In[8]:


def recurse_likelihood(node, site_i, n_states):
	if node.is_leaf():
		node.max_parsimony.fill(inf) # reset the leaf likelihoods
		leaf_state = node.sequence[site_i]
		node.max_parsimony[leaf_state] = 0
	else:
		left_child, right_child = node.get_children()
		recurse_likelihood(left_child, site_i, n_states)
		recurse_likelihood(right_child, site_i, n_states)

		for node_state in range(n_states):
			this = numpy.zeros(n_states) 
			for i in range(n_states):
			# Find min-cost change from node on the left
				min_left = inf
				for j in range(n_states):
					if has_sequence(left_child):
						this_cost = cost_matrix[i][j] + left_child.sequence[j]
						min_left = min(min_left, this_cost)
				# Find min-cost change from node on the right
				min_right = inf
				for k in range(n_states):
					if has_sequence(right_child):
						this_cost = cost_matrix[i][k] + right_child.sequence[k]
						min_right = min(min_right, this_cost)
			node.max_parsimony[node_state] = min_left + min_right
			return this


# In[6]:


# nucleotides, obviously
alphabet = "ACGT" # A = 0, C = 1, G = 2, T = 3
n_states = len(alphabet)


# this script requires a newick tree file and fasta sequence file, and
# the paths to those two files are given as arguments to this script

tree_path = "./example-tree.newick"
root_node = read_newick(tree_path)

msa_path = "./example-msa.fasta"
taxa, alignment = read_fasta(msa_path, alphabet)
site_count = len(alignment[0])

# the number of taxa, and the number of nodes in a rooted phylogeny with that
# number of taxa
n_taxa = len(taxa)
n_nodes = n_taxa + n_taxa - 1

# add sequences to leaves
for node in root_node.traverse():
	# initialize a vector of partial likelihoods that we can reuse for each site
	node.max_parsimony = numpy.zeros(n_states)

	if node.is_leaf():
		taxon = node.name
		taxon_i = taxa.index(taxon)
		node.sequence = alignment[taxon_i]


# In[9]:


# this will be the total likelihood of all sites
parsimony_score = 0.0
#min_parsiomny = min(root_parsimony)

for site_i in range(site_count):
	recurse_likelihood(root_node, site_i, n_states)

	# need to multiply the partial likelihoods by the stationary frequencies
	# which for Jukes-Cantor is 1/4 for all states
	parsimony_score += min(root_node.max_parsimony)
tree_filename = os.path.split(tree_path)[1]
msa_filename = os.path.split(msa_path)[1]

tree_name = os.path.splitext(tree_filename)[0]
msa_name = os.path.splitext(msa_filename)[0]

print("The log likelihood P(%s|%s) = %f" % (msa_name, tree_name, parsimony_score))


# In[ ]:




