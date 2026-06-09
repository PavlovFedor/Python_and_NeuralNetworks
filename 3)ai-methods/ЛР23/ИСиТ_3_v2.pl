%print_houses.
%Примеры:
%find_house(pet, cat, House).
%find_house(nationality, german, House).
%find_house(drink, beer, House).
%
%
%
% Определение структуры дома:
% house(Color, Nationality, Drink, Cigarette, Pet)

% Основное правило для решения задачи
solve(Houses) :-
    % Создаем список из 5 домов
    Houses = [house(_, _, _, _, _), house(_, _, _, _, _), house(_, _, _, _, _), house(_, _, _, _, _), house(_, _, _, _, _)],
    
    % Англичанин живет в красном доме
    member(house(red, british, _, _, _), Houses),
    
    % Швед держит собаку
    member(house(_, swedish, _, _, dog), Houses),
    
    % Датчанин пьет чай
    member(house(_, danish, tea, _, _), Houses),
    
    % Зеленый дом находится слева от белого
    left_of(house(green, _, _, _, _), house(white, _, _, _, _), Houses),
    
    % В зеленом доме пьют кофе
    member(house(green, _, coffee, _, _), Houses),
    
    % Человек, который курит Pall Mall, разводит птиц
    member(house(_, _, _, pall_mall, bird), Houses),
    
    % В желтом доме курят Dunhill
    member(house(yellow, _, _, dunhill, _), Houses),
    
    % Норвежец живет в первом доме
    nth1(1, Houses, house(_, norwegian, _, _, _)),
    
    % Молоко пьют в среднем доме (дом 3)
    nth1(3, Houses, house(_, _, milk, _, _)),
    
    % Человек, который живет рядом с тем, кто курит Marlboro, пьет воду
    next_to(house(_, _, water, _, _), house(_, _, _, marlboro, _), Houses),
    
    % Человек, который курит Winfield, пьет пиво
    member(house(_, _, beer, winfield, _), Houses),
    
    % Немец курит Rothmans
    member(house(_, german, _, rothmans, _), Houses),
    
    % Человек, который курит Marlboro, живет рядом с тем, кто держит кошку
    next_to(house(_, _, _, marlboro, _), house(_, _, _, _, cat), Houses),
    
    % Человек, который держит лошадь, живет рядом с тем, кто курит Dunhill
    next_to(house(_, _, _, _, horse), house(_, _, _, dunhill, _), Houses),
    
    % Норвежец живет рядом с синим домом
    next_to(house(_, norwegian, _, _, _), house(blue, _, _, _, _), Houses),
    
    % Находим хозяина, который держит рыбу
    member(house(_, german, _, _, fish), Houses).

% Вспомогательные правила

% Проверка, что дом A находится слева от дома B
left_of(A, B, Houses) :-
    nth1(IndexA, Houses, A),
    nth1(IndexB, Houses, B),
    IndexB is IndexA + 1.

% Проверка, что два дома находятся рядом
next_to(A, B, Houses) :-
    left_of(A, B, Houses).
next_to(A, B, Houses) :-
    left_of(B, A, Houses).

% Поиск дома по параметру
find_house(Parameter, Value, House) :-
    solve(Houses), % Решаем задачу и получаем Houses
    member(House, Houses),
    (Parameter = color, House = house(Value, _, _, _, _);
     Parameter = nationality, House = house(_, Value, _, _, _);
     Parameter = drink, House = house(_, _, Value, _, _);
     Parameter = cigarette, House = house(_, _, _, Value, _);
     Parameter = pet, House = house(_, _, _, _, Value)).

% Правило для вывода всех домов (для удобства)
print_houses :-
    solve(Houses),
    forall(member(House, Houses), format('~w~n', [House])).