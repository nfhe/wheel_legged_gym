import math as m


l1 = 0.1
l2 = 0.165
l0_offset = 0.145
theta1 = 3.065
theta2 = -2.285

x = l1 * m.cos(theta1) + l2 * m.cos(theta1 + theta2) - l0_offset
y = l1 * m.sin(theta1) + l2 * m.sin(theta1 + theta2)
print(x, y)