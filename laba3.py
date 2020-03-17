import random
import numpy
from scipy.stats import t,f

x1_min = 15
x1_max = 45

x2_min = 15
x2_max = 50

x3_min = 15
x3_max = 30

xm_min = (x1_min + x2_min + x3_min) / 3
xm_max = (x1_max + x2_max + x3_max) / 3
y_min = 200 + xm_min
y_max = 200 + xm_max

xn = [[-1, -1, -1],
      [-1, 1, 1],
      [1, -1, 1],
      [1, 1, -1]]

x = [[x1_min, x2_min, x3_min],
     [x1_min, x2_max, x3_max],
     [x1_max, x2_min, x3_max],
     [x1_max, x2_max, x3_min]]

m = 2
y = [[random.randint(int(y_min), int(y_max)) for i in range(m)] for j in range(4)]


def table_student(prob, f3):
    x_vec = [i*0.0001 for i in range(int(5/0.0001))]
    par = 0.5 + prob/0.1*0.05
    for i in x_vec:
        if abs(t.cdf(i, f3) - par) < 0.000005:
            return i


def table_fisher(prob, d, f3):
    x_vec = [i*0.001 for i in range(int(10/0.001))]
    for i in x_vec:
        if abs(f.cdf(i, 4-d, f3)-prob) < 0.0001:
            return i


def kohren(dispersion, m):
    fisher = table_fisher(0.95, 1, (m - 1) * 4)
    gt = fisher/(fisher+(m-1)-2)
    return max(dispersion) / sum(dispersion) < gt


def student(dispersion_reproduction, m, y_mean, xn):
    dispersion_statistic_mark = (dispersion_reproduction / (4 * m)) ** 0.5

    beta = [1 / 4 * sum(y_mean[j] for j in range(4))]
    for i in range(3):
        b = 0
        for j in range(4):
            b += y_mean[j] * xn[j][i]
        beta.append(1 / 4 * b)

    t = []
    for i in beta:
        t.append(abs(i) / dispersion_statistic_mark)

    f3 = (m - 1) * 4

    return t[0] > table_student(0.95, f3), t[1] > table_student(0.95, f3), t[2] > table_student(0.95, f3), t[3] > table_student(0.95, f3)


def normalized_multiplier(x, y_mean):
    mx = [0, 0, 0]
    axx = [0, 0, 0]
    ax = [0, 0, 0]
    for i in range(3):
        for j in range(4):
            mx[i] += x[j][i]
            axx[i] += x[j][i] ** 2
            ax[i] += x[j][i] * y_mean[j]
        mx[i] /= 4
        axx[i] /= 4
        ax[i] /= 4
    
    my = sum(y_mean) / 4

    a12	= (x[0][0] * x[0][1] + x[1][0] * x[1][1] + x[2][0] * x[2][1] + x[3][0] * x[3][1]) / 4
    a13	= (x[0][0] * x[0][2] + x[1][0] * x[1][2] + x[2][0] * x[2][2] + x[3][0] * x[3][2]) / 4
    a23	= (x[0][1] * x[0][2] + x[1][1] * x[1][2] + x[2][1] * x[2][2] + x[3][1] * x[3][2]) / 4

    a = numpy.array([[1, *mx],
                     [mx[0], axx[0], a12, a13],
                     [mx[1], a12, axx[1], a23],
                     [mx[2], a13, a23, axx[2]]])
    c = numpy.array([my, *ax])
    b = numpy.linalg.solve(a, c)
    return b


def fisher(m, d, y_mean, yo, dispersion_reproduction):

    dispersion_ad = 0
    for i in range(4):
        dispersion_ad += (yo[i] - y_mean[i]) ** 2
        
    dispersion_ad = dispersion_ad * m / (4 - d)

    fp = dispersion_ad / dispersion_reproduction

    f3 = (m - 1) * 4
    
    return fp < table_fisher(0.95, d, f3)


while True:
    while True:
        if m > 8:
            print("Current m is more than max number in database of Student criterion. Please restart")
            exit(0)
            
        y_mean = []
        for i in range(4):
            y_mean.append(sum(y[i]) / m)
        
        dispersion = []
        for i in range(len(y)):
            dispersion.append(0)
            for j in range(m):
                dispersion[i] += (y_mean[i] - y[i][j]) ** 2
            dispersion[i] /= m

        dispersion_reproduction = sum(dispersion) / 4

        if kohren(dispersion, m):
            break
        else:
            m += 1
            for i in range(4):
                y[i].append(random.randint(int(y_min), int(y_max)))

    k = student(dispersion_reproduction, m, y_mean, xn)
    d = sum(k)
    
    b = normalized_multiplier(x, y_mean)
    b = [b[i] * k[i] for i in range(4)]

    yo = []
    for i in range(4):
        yo.append(b[0] + b[1] * x[i][0] + b[2] * x[i][1] + b[3] * x[i][2])
    
    if d == 4:
        m += 1
        for i in range(4):
            y[i].append(random.randint(int(y_min), int(y_max)))
        
    elif fisher(m, d, y_mean, yo, dispersion_reproduction):
        break
    else:
        m += 1
        for i in range(4):
            y[i].append(random.randint(int(y_min), int(y_max)))

#console output
print("\n| № | X1 | X2 | X3 |", end="")

for i in range(m):
    print(" Yi{:d} |".format(i+1), end="")

print()
for i in range(4):
    print("| {:1d} | {:2d} | {:2d} | {:2d} |".format(i+1, *x[i]), end="")
    for j in y[i]:
        print(" {:3d} |".format(j), end="")
    print()

print("\nUsing G(Kohren) - criterion, current dispersion is uniform.")
print("Usind T(Student) - criterion, relevance of \n\tb0 is {}, b1 - {} b2 - {}, b3 - {}".format(*k))
print("Using F(Fisher) - criterion, current regerecy equation is adequate.")

print("\n\tLinear regrecy equation:\tY = {:.2f}".format(b[0]), end="")
for i in range(1,4):
    if b[i] != 0:
        print(" + {0:.2f}".format(b[i]) + "*X" + str(i), end="")

print("\n\nControl result:")
for i in range(4):
    print("\t\t\t\tYs{:d}\t= {:.2f}\n\t b0 + b1*X1 + b2*X2 + b3*X3\t= {:.2f}"
          .format(i+1, y_mean[i], b[0] + b[1] * x[i][0] + b[2] * x[i][1]))
    print()
