for line in open("kinship.nl", "r").readlines():
	line = line.strip()
	line = line[:-1]
	relation, remaining = line.split('(')
	head, remaining = remaining.split(',')
	tail = remaining[:-1]
	print(head, relation, tail, sep = "\t")