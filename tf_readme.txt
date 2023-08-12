####################################################################################

The transcription factor (tf) database has the following structure: [154, 7, 1503]

we collected 7 properties for each amino acid out of 154 different tfs

####################################################################################

Properties order and description:

############################################
1. Amino acid sequence
a vector of 1503 values.
the max length of the amino acid sequence is 1502.
the last value of every entry is its DBD class token.
if an entry has less than 1502 amino acids, the gap is filled with (-1) values.

The amino acid token is based on the following dicitonary:
 'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20

The DBD class token is based on the follwoing dictionary:
    'Abf1': 1, 'Sox': 2, 'bZIP': 3, 'ZF': 4, 'AFT': 5, 'Cluster': 6,
    'GATA': 7, 'Myb': 8, 'bHLH': 9, 'missing': 10, 'Forkhead': 11, 'Gcr1': 12,
    'fist': 13, 'bZip': 14, 'CBF': 15, 'HSF': 16, 'APSES': 17, 'MADS': 18,
    'GCR1': 19, 'SAP': 20, 'Homeodomain': 21, 'MADF': 22, 'RFX': 23,
    'Ste': 23, 'AT hook': 24, 'TEA': 25, 'VHR1': 26

#############################################
2. Disorder-score
A vector of 1503 values.
every value represents the probability of this amino acid being part of an IDR.
gets values from 0 to 1.
the 1503th value has no meaning and its set to 0.

#############################################
3. Molecular weight of the amino acid
A vector of 1503 values.
represents the molecular weight of the entire amino acid. this parameter indicates the entire amino acid size.
just like in the disorder-score, the 1503th value is set to 0 and has no meaning.

#############################################
4. Residue's molecular weight
just like parameter #3, but this time only for the amino acid's residue
just like in the disorder-score, the 1503th value is set to 0 and has no meaning.

#############################################
5. pKa of the amino acid

the negative of the logarithm of the dissociation constant for the -COOH group in the amino acid
just like in the disorder-score, the 1503th value is set to 0 and has no meaning.

#############################################
6. pKb of the amino acid

the negative of the logarithm of the dissociation constant for the -NH3 group in the amino acid
just like in the disorder-score, the 1503th value is set to 0 and has no meaning.

#############################################
7. Isoelectric point of the amino acid

the pH in which this amino acids holds a net charge equal to 0
just like in the disorder-score, the 1503th value is set to 0 and has no meaning.

