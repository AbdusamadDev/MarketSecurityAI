first = list(range(0, 270, 2))[::-1]
second = list(range(-1, 270, 2))[::-1]
third = list(zip(first, second))
fourth = []
for i, j in third:
    if (third.index((i, j)) % 2) == 1:
        fourth.append(j)
        fourth.append(i)
print(fourth)
