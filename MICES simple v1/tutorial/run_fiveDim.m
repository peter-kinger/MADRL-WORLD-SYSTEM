% 定义时间范围
tspan = [0 10];
% 定义初始条件
y0 = [1; 0; 0; 0; 0];

a = -2;
b = 3;

% 使用 ode45 求解微分方程
% 后续可以考虑简化代码
% [t, y] = ode45(@fiveDimODE, tspan, y0);
[t, y] = ode45(@(t,y) fiveDimODE(y, a,b), tspan, y0);

% 绘制结果
figure;
plot(t, y(:, 1), '-o', 'DisplayName', 'y1');
hold on;
plot(t, y(:, 2), '-+', 'DisplayName', 'y2');
plot(t, y(:, 3), '-*', 'DisplayName', 'y3');
plot(t, y(:, 4), '-s', 'DisplayName', 'y4');
plot(t, y(:, 5), '-d', 'DisplayName', 'y5');
hold off;
xlabel('Time');
ylabel('y');
title('Solution of 5-Dimensional ODE System');
legend;
