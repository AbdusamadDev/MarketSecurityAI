def capitalize(name: str):
    capital = [chr(i) for i in range(65, 92)]
    lowered = [chr(j) for j in range(97, 123)]
    a = name[0]
    if a in lowered:
        a = capital[lowered.index(a)]
    b = name[1:]
    return a + b


print(capitalize("salom"))
