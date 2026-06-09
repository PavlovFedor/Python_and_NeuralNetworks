% start(Solution).
% Определим дом как структуру: house(Color, Nationality, Drink, Cigarette, Pet)
solve(Solution) :-
    % Создаем список из 5 домов
    Solution = [house(_, _, _, _, _), house(_, _, _, _, _), house(_, _, _, _, _), house(_, _, _, _, _), house(_, _, _, _, _)],

    % Норвежец живет в первом доме
    nth1(1, Solution, house(_, norwegian, _, _, _)),

    % Англичанин живет в красном доме
    member(house(red, english, _, _, _), Solution),

    % Зеленый дом находится слева от белого
    left_of(house(green, _, _, _, _), house(white, _, _, _, _), Solution),

    % Датчанин пьет чай
    member(house(_, dane, tea, _, _), Solution),

    % Тот, кто курит Marlboro, живет рядом с тем, кто выращивает кошек
    next_to(house(_, _, _, marlboro, _), house(_, _, _, _, cats), Solution),

    % Тот, кто живет в желтом доме, курит Dunhill
    member(house(yellow, _, _, dunhill, _), Solution),

    % Немец курит Rothmans
    member(house(_, german, _, rothmans, _), Solution),

    % Тот, кто живет в центре, пьет молоко
    nth1(3, Solution, house(_, _, milk, _, _)),

    % Сосед того, кто курит Marlboro, пьет воду
    next_to(house(_, _, _, marlboro, _), house(_, _, water, _, _), Solution),

    % Тот, кто курит Pall Mall, выращивает птиц
    member(house(_, _, _, pallmall, birds), Solution),

    % Швед выращивает собак
    member(house(_, swede, _, _, dogs), Solution),

    % Норвежец живет рядом с синим домом
    next_to(house(_, norwegian, _, _, _), house(blue, _, _, _, _), Solution),

    % Тот, кто выращивает лошадей, живет в синем доме
    member(house(blue, _, _, _, horses), Solution),

    % Тот, кто курит Winfield, пьет пиво
    member(house(_, _, beer, winfield, _), Solution),

    % В зеленом доме пьют кофе
    member(house(green, _, coffee, _, _), Solution).

% Предикат для проверки, что один дом находится слева от другого
left_of(A, B, [A, B | _]).
left_of(A, B, [_ | Tail]) :- left_of(A, B, Tail).

% Предикат для проверки, что два дома находятся рядом
next_to(A, B, Solution) :-
    (left_of(A, B, Solution); left_of(B, A, Solution)).

% Запуск решения
start(Solution) :-
    solve(Solution),
    % Выводим результат
    write('Answer:'), nl,
    print_houses(Solution).

% Предикат для вывода домов
print_houses([]).
print_houses([House | Tail]) :-
    write(House), nl,
    print_houses(Tail).