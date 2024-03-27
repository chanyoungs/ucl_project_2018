s = ''

for n in range(10):
    for m in range(10):
        s += 'cp data{0}/3dSprites{1:06d}.png Check/{0}_{2}.png \n'.format(n, m*10000, m)
# print(s)

with open("Check.txt", "w") as text_file:
    text_file.write(s)
