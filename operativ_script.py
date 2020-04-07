import itertools

z = [ "quarante", "vingt", "un", "dix", "et", "quatre", "sept", "onze", "neuf", "quinze", "soixante"]

for x1, x2, x3, x4 in itertools.product(range(11),range(11),range(11),range(11)):
    print(z[x1], z[x2], z[x3], z[x4], end='', sep='')
