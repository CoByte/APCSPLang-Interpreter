import operator as oper_built_in
from enum import Enum, auto
from functools import wraps
from copy import copy
from random import randint
import tkinter as tk
from turtle import RawTurtle, TurtleScreen


###############################################################################
#                                                                             #
#  LEXER                                                                      #
#                                                                             #
###############################################################################


class TokenType(Enum):
    INTEGER = auto(),
    FLOAT = auto(),
    BOOLEAN = auto(),
    STRING = auto(),
    EOF = auto(),
    ADD = auto(),
    SUB = auto(),
    MUL = auto(),
    DIV = auto(),
    MOD = auto(),
    LPAREN = auto(),
    RPAREN = auto(),
    ID = auto(),
    ASSIGN = auto(),
    NEWLINE = auto(),
    PROCEDURE = auto(),
    LBRACE = auto(),
    RBRACE = auto(),
    COMMA = auto(),
    AND = auto(),
    OR = auto(),
    NOT = auto(),
    NEQUAL = auto(),
    GEQUAL = auto(),
    LEQUAL = auto(),
    GREATER = auto(),
    LESS = auto(),
    RETURN = auto(),
    IF = auto(),
    ELSE = auto(),
    REPEAT = auto(),
    TIMES = auto(),
    UNTIL = auto(),
    EQUAL = auto(),
    LBRACKET = auto(),
    RBRACKET = auto(),
    FOR = auto(),
    EACH = auto(),
    IN = auto()

class Token(object):
    def __init__(self, type: TokenType, value=None):
        self.type = type
        self.value = self.type if value is None else value

    def __str__(self):
        # String representation of the class instance.

        # Examples:
        #     Token(INTEGER, 3)
        #     Token(PLUS '+')
        return f"Token({self.type.name}, {repr(self.value)})"

    def __repr__(self):
        return self.__str__()


SYMBOL_MAPPING = {
    "+": TokenType.ADD,
    "-": TokenType.SUB,
    "*": TokenType.MUL,
    "/": TokenType.DIV,
    "(": TokenType.LPAREN,
    ")": TokenType.RPAREN,
    "\n": TokenType.NEWLINE,
    "{": TokenType.LBRACE,
    "}": TokenType.RBRACE,
    ",": TokenType.COMMA,
    ">": TokenType.GREATER,
    "<": TokenType.LESS,
    "=": TokenType.EQUAL,
    "[": TokenType.LBRACKET,
    "]": TokenType.RBRACKET,
    "≠": TokenType.LEQUAL,
    "≥": TokenType.GEQUAL,
    "≤": TokenType.NEQUAL
}

RESERVED_KEYWORDS = {
    "PROCEDURE": Token(TokenType.PROCEDURE),
    "AND": Token(TokenType.AND),
    "OR": Token(TokenType.OR),
    "NOT": Token(TokenType.NOT),
    "TRUE": Token(TokenType.BOOLEAN, True),
    "FALSE": Token(TokenType.BOOLEAN, False),
    "RETURN": Token(TokenType.RETURN),
    "IF": Token(TokenType.IF),
    "ELSE": Token(TokenType.ELSE),
    "REPEAT": Token(TokenType.REPEAT),
    "TIMES": Token(TokenType.TIMES),
    "UNTIL": Token(TokenType.UNTIL),
    "MOD": Token(TokenType.MOD),
    "FOR": Token(TokenType.FOR),
    "EACH": Token(TokenType.EACH),
    "IN": Token(TokenType.IN)
}


class Lexer(object):
    def __init__(self, text):
        # client string input, e.g. "3 + 5", "12 - 5", etc
        self.text = text

        # self.pos is an index into self.text
        self.pos = 0
        self.seen_eof = False

    # for accessing the current_char
    @property
    def current_char(self):
        if self.pos >= len(self.text):
            return None

        return self.text[self.pos]

    # for accessing the next char
    @property
    def next_char(self):
        if self.pos + 1 >= len(self.text):
            return None

        return self.text[self.pos + 1]

    # skips over whitespace
    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace() and self.current_char != "\n":
            self.pos += 1

    # gloms together digits
    def str_number(self) -> str:
        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.pos += 1

        return result

    # gloms numbers together
    # works with both integers and floats
    def number(self) -> Token:
        result = self.str_number()

        if self.current_char != ".":
            return Token(TokenType.INTEGER, int(result))

        self.pos += 1
        result = float(result + "." + self.str_number())
        return Token(TokenType.FLOAT, result)

    def string(self) -> Token:
        self.pos += 1
        result = ""
        while self.current_char != '"':
            if self.current_char is None:
                raise Exception(f"Error parsing input: string left unclosed")
            
            result += self.current_char
            self.pos += 1
        self.pos += 1

        return Token(TokenType.STRING, result)

    # gloms ids
    def id(self):
        result = ""
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char =="_"):
            result += self.current_char
            self.pos += 1

        return RESERVED_KEYWORDS.get(result, Token(TokenType.ID, result))

    def get_next_token(self) -> Token:
        # Lexical analyzer (also known as scanner or tokenizer)

        # This method is responsible for breaking a sentence
        # apart into tokens.
        while self.current_char is not None:

            if self.current_char.isspace() and self.current_char != "\n":
                self.skip_whitespace()
                continue

            if self.current_char.isdigit():
                return self.number()

            if self.current_char == '"':
                return self.string()

            if self.current_char.isalpha():
                return self.id()

            if self.current_char == "<" and self.next_char == "=":
                self.pos += 2
                return Token(TokenType.ASSIGN)

            if self.current_char == "=" and self.next_char == "<":
                self.pos += 2
                return Token(TokenType.LEQUAL)

            if self.current_char == "=" and self.next_char == ">":
                self.pos += 2
                return Token(TokenType.GEQUAL)

            if self.current_char == "\\" and self.next_char == "=":
                self.pos += 2
                return Token(TokenType.NEQUAL)

            if self.current_char in SYMBOL_MAPPING:
                symbol = self.current_char
                self.pos += 1
                symbol_type = SYMBOL_MAPPING[symbol]
                return Token(symbol_type)

            raise Exception(f'Error parsing input: {self.current_char}')

        return Token(TokenType.EOF)

    def __iter__(self):
        self.pos = 0
        self.seen_eof = False
        return self

    def __next__(self):
        token = self.get_next_token()

        if token.type is TokenType.EOF:
            if self.seen_eof:
                raise StopIteration
            self.seen_eof = True

        return token


class Tokenized:
    def __init__(self, lexer: Lexer):
        self.lexer = lexer
        self.tokens = list(lexer)

        self.pos = 0

    @property
    def current_token(self):
        if self.pos >= len(self.tokens):
            return None

        return self.tokens[self.pos]

    @property
    def next_token(self):
        if self.pos + 1 >= len(self.tokens):
            return None

        return self.tokens[self.pos + 1]

    def inc_token(self):
        self.pos += 1


###############################################################################
#                                                                             #
#  PARSER                                                                     #
#                                                                             #
###############################################################################


class Node:
    pass


class Block(Node):
    def __init__(self, children):
        self.children = children


class Return(Node):
    def __init__(self, value):
        self.value = value


class Assign(Node):
    def __init__(self, var, value):
        self.var = var
        self.value = value


class Procedure(Node):
    def __init__(self, args: list, block: Block):
        self.args = args
        self.block = block


class IfStatement(Node):
    def __init__(self, condition, true_block, else_block):
        self.condition = condition
        self.true_block = true_block
        self.else_block = else_block


class IncrementLoop(Node):
    def __init__(self, value, block):
        self.value = value
        self.block = block


class UntilLoop(Node):
    def __init__(self, condition, block):
        self.condition = condition
        self.block = block


class ForEachLoop(Node):
    def __init__(self, id, list, block):
        self.id = id
        self.list = list
        self.block = block


class NoOp(Node):
    pass


class BinOp(Node):
    def __init__(self, left: Node, oper: Token, right: Node):
        self.left = left
        self.oper = oper
        self.right = right


class UnaryOp(Node):
    def __init__(self, oper: Token, expr: Node):
        self.oper = oper
        self.expr = expr


class Primative(Node):
    def __init__(self, token: Token):
        self.token = token
        self.value = token.value


class List(Node):
    def __init__(self, items):
        self.items = items


class ListAccess(Node):
    def __init__(self, list, index):
        self.list = list
        self.index = index


class Var(Node):
    def __init__(self, token):
        self.token = token
        self.value = token.value


class Call(Node):
    def __init__(self, var: Var, args: list):
        self.var = var
        self.args = args


def termlike(rule_name, *tokens):
    def inner(func):
        @wraps(func)
        def wrapper(obj):
            rule_method = getattr(obj, rule_name)

            node = rule_method()

            while obj.current_token.type in tokens:
                oper = obj.eat(*tokens)
                right = rule_method()

                node = BinOp(node, oper, right)

            return node
        return wrapper
    return inner

class Parser:
    def __init__(self, lexer: Lexer):
        self.lexer = lexer
        self.tokens = Tokenized(lexer)

    @property
    def current_token(self) -> Token:
        return self.tokens.current_token

    @property
    def next_token(self) -> Token:
        return self.tokens.next_token

    def error(self, token):
        raise Exception(f"Invalid syntax on token {token}")

    def eat(self, *type: TokenType) -> Token:
        if self.current_token.type in type:
            old_token = self.current_token
            self.tokens.inc_token()
            return old_token

        raise SyntaxError(f"Expected token of type {[t.name for t in type]}, found {self.current_token.type.name}")

    def eat_newlines(self):
        while self.current_token.type is TokenType.NEWLINE:
            self.eat(TokenType.NEWLINE)

    ### GRAMMER METHODS ###

    def program(self):
        if self.current_token.type is TokenType.EOF:
            return NoOp()

        return self.block()

    def block(self):
        statements = [self.statement()]

        while self.current_token.type is TokenType.NEWLINE:
            self.eat(TokenType.NEWLINE)

            if self.current_token.type in (TokenType.RBRACE, TokenType.EOF):
                break

            statements.append(self.statement())

        return Block(statements)

    def statement(self):
        token_type = self.current_token.type

        if token_type is TokenType.RETURN:
            return self.return_statement()

        if token_type is TokenType.ID and self.next_token.type is TokenType.LPAREN:
            return self.expr()

        if token_type in (TokenType.ID, TokenType.PROCEDURE):
            return self.assignment_statement()

        if token_type is TokenType.IF:
            return self.if_statement()

        if token_type is TokenType.REPEAT:
            return self.repeat_statement()

        if token_type is TokenType.FOR:
            return self.for_each_loop()

        if token_type is not TokenType.NEWLINE:
            return self.expr()

        return self.empty()

    def return_statement(self):
        self.eat(TokenType.RETURN)
        self.eat(TokenType.LPAREN)
        value = self.expr()
        self.eat(TokenType.RPAREN)

        return Return(value)

    def assignment_statement(self):
        if self.current_token.type is TokenType.PROCEDURE:
            self.eat(TokenType.PROCEDURE)
            name = Var(self.eat(TokenType.ID))
            value = self.procedure_definition()

        else:
            name = self.variable()
            self.eat(TokenType.ASSIGN)
            value = self.expr()

        return Assign(name, value)

    def procedure_definition(self):
        self.eat(TokenType.LPAREN)

        args = []
        if self.current_token.type is TokenType.ID:
            args.append(self.eat(TokenType.ID))

            while self.current_token.type is TokenType.COMMA:
                self.eat(TokenType.COMMA)
                args.append(self.eat(TokenType.ID))

        self.eat(TokenType.RPAREN)
        self.eat_newlines()
        self.eat(TokenType.LBRACE)
        block = self.block()
        self.eat(TokenType.RBRACE)

        return Procedure(args, block)

    def if_statement(self):
        self.eat(TokenType.IF)
        self.eat(TokenType.LPAREN)
        condition = self.expr()
        self.eat(TokenType.RPAREN)
        self.eat_newlines()
        self.eat(TokenType.LBRACE)
        true_block = self.block()
        self.eat(TokenType.RBRACE)

        while self.current_token.type is TokenType.NEWLINE:
            if self.next_token.type is TokenType.NEWLINE:
                self.eat(TokenType.NEWLINE)
                continue

            if self.next_token.type is TokenType.ELSE:
                self.eat(TokenType.NEWLINE)
                break

            return IfStatement(condition, true_block, None)

        else_block = None
        if self.current_token.type is TokenType.ELSE:
            self.eat(TokenType.ELSE)
            self.eat_newlines()
            self.eat(TokenType.LBRACE)
            else_block = self.block()
            self.eat(TokenType.RBRACE)

        return IfStatement(condition, true_block, else_block)

    def repeat_statement(self):
        self.eat(TokenType.REPEAT)

        if self.current_token.type is TokenType.UNTIL:
            return self.until_loop()

        return self.increment_loop()

    def increment_loop(self):
        value = self.expr()
        self.eat(TokenType.TIMES)
        self.eat_newlines()
        self.eat(TokenType.LBRACE)
        block = self.block()
        self.eat(TokenType.RBRACE)

        return IncrementLoop(value, block)


    def until_loop(self):
        self.eat(TokenType.UNTIL)
        self.eat(TokenType.LPAREN)
        condition = self.expr()
        self.eat(TokenType.RPAREN)
        self.eat_newlines()
        self.eat(TokenType.LBRACE)
        block = self.block()
        self.eat(TokenType.RBRACE)

        return UntilLoop(condition, block)

    def for_each_loop(self):
        self.eat(TokenType.FOR)
        self.eat(TokenType.EACH)
        id = self.eat(TokenType.ID)
        self.eat(TokenType.IN)
        _list = self.expr()
        self.eat_newlines()
        self.eat(TokenType.LBRACE)
        block = self.block()
        self.eat(TokenType.RBRACE)

        return ForEachLoop(id, _list, block)

    def empty(self):
        return NoOp()

    @termlike("term_compare", TokenType.AND, TokenType.OR)
    def expr(self):
        pass

    @termlike(
        "term_addsub", 
        TokenType.GREATER, 
        TokenType.LESS,
        TokenType.GEQUAL,
        TokenType.LEQUAL,
        TokenType.EQUAL,
        TokenType.NEQUAL
    )
    def term_compare(self):
        pass

    @termlike(
        "term_muldiv", 
        TokenType.ADD, 
        TokenType.SUB,
        TokenType.MOD
    )
    def term_addsub(self):
        pass

    @termlike("factor", TokenType.MUL, TokenType.DIV)
    def term_muldiv(self):
        pass

    def factor(self):
        token_type = self.current_token.type

        if token_type in (TokenType.ADD, TokenType.SUB, TokenType.NOT):
            return UnaryOp(self.eat(token_type), self.factor())

        if token_type in (TokenType.INTEGER, TokenType.FLOAT, TokenType.BOOLEAN, TokenType.STRING):
            return Primative(self.eat(token_type))

        if token_type is TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            node = self.expr()
            self.eat(TokenType.RPAREN)
            return node

        if token_type in (TokenType.ID, TokenType.LBRACKET):
            return self.variable()

        self.error(token_type)
        
    def variable(self):
        if self.current_token.type is TokenType.LBRACKET:
            self.eat(TokenType.LBRACKET)

            items = []
            if self.current_token.type is not TokenType.RBRACKET:
                self.eat_newlines()
                items.append(self.expr())
                
                while self.current_token.type is TokenType.COMMA:
                    self.eat(TokenType.COMMA)
                    self.eat_newlines()
                    items.append(self.expr())
            self.eat_newlines()
            self.eat(TokenType.RBRACKET)
            
            var = List(items)

        else:
            var = Var(self.eat(TokenType.ID))

        while self.current_token.type in (TokenType.LPAREN, TokenType.LBRACKET):
            if self.current_token.type is TokenType.LPAREN:
                self.eat(TokenType.LPAREN)

                args = []
                if self.current_token.type is not TokenType.RPAREN:
                    self.eat_newlines()
                    args.append(self.expr())
                    while self.current_token.type is TokenType.COMMA:
                        self.eat(TokenType.COMMA)
                        self.eat_newlines()
                        args.append(self.expr())
                self.eat_newlines()
                self.eat(TokenType.RPAREN)

                var = Call(var, args)

            else:
                self.eat(TokenType.LBRACKET)
                self.eat_newlines()
                index = self.expr()
                self.eat_newlines()
                self.eat(TokenType.RBRACKET)

                var = ListAccess(var, index)

        return var                

    def parse(self):
        parsed = self.program()
        if self.current_token.type is not TokenType.EOF:
            self.error(self.current_token)

        return parsed


###############################################################################
#                                                                             #
#  INTERPRETER                                                                #
#                                                                             #
###############################################################################


OKCYAN = '\033[96m'
ENDC = '\033[0m'

class CallStack(dict):
    def __init__(self):
        self.frames = []

    def push(self, frame):
        self.frames.append(frame)

    def pop(self):
        return self.frames.pop()

    def peek(self):
        return self.frames[-1]

    def __getitem__(self, key):
        for frame in reversed(self.frames):
            if key in frame:
                return frame[key]
        
        return None

    def __setitem__(self, key, value):
        self.peek()[key] = value

    def __str__(self) -> str:
        output = ""
        for count, frame in reversed(list(enumerate(self.frames))):
            output += f"Frame {count}\n"
            if len(frame.keys()) == 0:
                output += "  None\n"
            for k, v in frame.items():
                output += f"  {k}: {v}\n"
        return output

    __repr__ = __str__


class MemType(Enum):
    INTEGER = auto(),
    FLOAT = auto(),
    BOOLEAN = auto(),
    STRING = auto(),
    PROCEDURE = auto(),
    LIST = auto(),
    NONE = auto(),

    # special types
    EXTERNAL_PROCEDURE = auto(),
    RETURN = auto()


PRIMATIVE_TO_MEMTYPE = {
    TokenType.INTEGER: MemType.INTEGER,
    TokenType.FLOAT: MemType.FLOAT,
    TokenType.BOOLEAN: MemType.BOOLEAN,
    TokenType.STRING: MemType.STRING
}


NUM_TYPES = (MemType.INTEGER, MemType.FLOAT)        


class MemEntry:
    def __init__(self, value, type: MemType):
        self.value = value
        self.type = type

    def __str__(self):
        return f"{self.type.name}: {self.value.__str__()}"

    __repr__ = __str__


def numerical_op(oper):
    def inner(*entries: MemEntry):
        args = [entry.value for entry in entries]
        types = [entry.type for entry in entries]

        value = oper(*args)
        
        if MemType.FLOAT in types:
            type = MemType.FLOAT
        else:
            type = MemType.INTEGER
            value = int(value)

        return MemEntry(value, type)

    return inner


class BinaryOperator:
    def __init__(self, func, *valid_type):
        self.func = func
        self.valid_type = valid_type

    def __call__(self, left: MemEntry, right: MemEntry):
        if left.type in self.valid_type and right.type in self.valid_type:
            return self.func(left, right)

        raise TypeError()


def equal_binop(left: MemEntry, right: MemEntry):
    if left.type is right.type:
        return MemEntry(left.value == right.value, MemType.BOOLEAN)

    raise TypeError()


BINARY_OPERATORS = {
    TokenType.ADD: BinaryOperator(
        numerical_op(oper_built_in.add),
        *NUM_TYPES
    ),
    TokenType.SUB: BinaryOperator(
        numerical_op(oper_built_in.sub),
        *NUM_TYPES
    ),
    TokenType.MUL: BinaryOperator(
        numerical_op(oper_built_in.mul),
        *NUM_TYPES
    ),
    TokenType.DIV: BinaryOperator(
        numerical_op(oper_built_in.truediv),
        *NUM_TYPES
    ),
    TokenType.MOD: BinaryOperator(
        lambda a, b: MemEntry(a.value % b.value, MemType.INTEGER),
        MemType.INTEGER
    ),
    TokenType.GREATER: BinaryOperator(
        lambda a, b: MemEntry(a.value > b.value, MemType.BOOLEAN),
        *NUM_TYPES
    ),
    TokenType.LESS: BinaryOperator(
        lambda a, b: MemEntry(a.value < b.value, MemType.BOOLEAN),
        *NUM_TYPES
    ),
    TokenType.GEQUAL: BinaryOperator(
        lambda a, b: MemEntry(a.value >= b.value, MemType.BOOLEAN),
        *NUM_TYPES
    ),
    TokenType.LEQUAL: BinaryOperator(
        lambda a, b: MemEntry(a.value <= b.value, MemType.BOOLEAN),
        *NUM_TYPES
    ),
    TokenType.AND: BinaryOperator(
        lambda a, b: MemEntry(a.value and b.value, MemType.BOOLEAN),
        MemType.BOOLEAN
    ),
    TokenType.OR: BinaryOperator(
        lambda a, b: MemEntry(a.value or b.value, MemType.BOOLEAN),
        MemType.BOOLEAN
    ),
    TokenType.EQUAL: equal_binop,
    TokenType.NEQUAL: lambda a, b: MemEntry(not equal_binop(a, b).value, MemType.BOOLEAN)
}


class UnaryOperator:
    def __init__(self, func, type: MemType):
        self.func = func
        self.type = type

    def __call__(self, arg):
        if arg.type in self.type:
            return self.func(arg)

        raise TypeError()


UNARY_OPERATORS = {
    TokenType.ADD: UnaryOperator(
        numerical_op(lambda a: a),
        NUM_TYPES
    ),
    TokenType.SUB: UnaryOperator(
        numerical_op(lambda a: -a),
        NUM_TYPES
    ),
    TokenType.NOT: UnaryOperator(
        lambda a: MemEntry(not a.value, MemType.BOOLEAN),
        [MemType.BOOLEAN]
    )
}


class Robot:
    instance = None
    has_initialized = False

    def __new__(cls, *args, **kwargs):
        if Robot.instance is None:
            Robot.instance = super(Robot, cls).__new__(cls, *args, **kwargs)
        return Robot.instance

    def __init__(self):
        if Robot.has_initialized:
            return

        self.root = tk.Tk()

        self.canvas = None
        self.screen = None
        self.turtle = None

        self.width = None
        self.height = None

        self.covered_boxes = []

        Robot.has_initialized = True

    def setup(self, x, y):
        self.width = x * 50
        self.height = y * 50

        if self.canvas is not None:
            self.canvas.destroy()
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height)
        self.canvas.pack()
        self.screen = TurtleScreen(self.canvas)
        self.turtle = RawTurtle(self.canvas)

        self.turtle.ht()
        self.turtle.shape("turtle")
        self.turtle.shapesize(*([1.5]*3))

        for i in range(50, self.width, 50):
            self.goto_noline(i, 0)
            self.goto_line(i, self.height)

        for i in range(50, self.height, 50):
            self.goto_noline(0, i)
            self.goto_line(self.width, i)

        self.set_position(0, 0)
        self.turtle.showturtle()


    def goto_line(self, x, y):
        self.turtle.speed(0)
        self.turtle.pd()
        self.turtle.goto(x - (self.width / 2), y - (self.height / 2))
        self.turtle.pu()
        self.turtle.speed(1)

    def goto_noline(self, x, y):
        self.turtle.speed(0)
        self.turtle.pu()
        self.turtle.goto(x - (self.width / 2), y - (self.height / 2))
        self.turtle.speed(1)

    def box(self, x, y):
        self.covered_boxes.append((x, y))

        x *= 50
        y *= 50

        self.turtle.hideturtle()
        turtle_current_pos = self.pos()

        self.goto_noline(x, y)

        self.turtle.speed(0)
        self.turtle.fillcolor("grey")
        self.turtle.begin_fill()
        for i in range(4):
            self.turtle.fd(50)
            self.turtle.left(90)
        self.turtle.end_fill()
        self.turtle.fillcolor("black")

        self.goto_noline(*turtle_current_pos)
        self.turtle.showturtle()

    def set_position(self, x, y):
        self.goto_noline(x*50 + 25, y*50 + 25)

    def pos(self):
        x, y = self.turtle.pos() + (self.width / 2, self.height / 2)
        return round(x), round(y)

    def rotate_left(self):
        self.turtle.left(90)

    def rotate_right(self):
        self.turtle.right(90)

    def forward(self):
        self.turtle.fd(50)

        pos = self.pos()
       
        if pos[0] < 0 or self.height < pos[0] or pos[1] < 0 or self.height < pos[1]:
            raise Exception("Robot out of bounds")

        turtle_x = int((pos[0]-25)/50)
        turtle_y = int((pos[1]-25)/50)

        if (turtle_x, turtle_y) in self.covered_boxes:
            raise Exception("Robot has entered box")

    def can_move(self, direction):
        headings = {
            "right": (1,0),
            "up": (0,1),
            "left": (-1,0),
            "down": (0,-1)
        }

        if direction not in headings:
            raise Exception("Invalid direction")

        tx, ty = self.pos()
        tx = (tx-25)//50 + headings[direction][0]
        ty = (ty-25)//50 + headings[direction][1]
        test_pos = (tx, ty)

        if test_pos[0] < 0 or self.width // 50 <= test_pos[0] or test_pos[1] < 0 or self.height // 50 <= test_pos[1]:
            return False

        if test_pos in self.covered_boxes:
            return False

        return True

class ExternalProcedure(Node):
    def __init__(self, name, func):
        self.name = name
        self.func = func

    @staticmethod
    def new(name, func):
        return MemEntry(ExternalProcedure(name, func), MemType.EXTERNAL_PROCEDURE)


def auto_external(*types):
    def wrapped(func):
        def inner(args):
            if len(types) != len(args):
                raise TypeError(f"Procedure {func.__name__} requires {len(types)} arguments, but {len(args)} were given")

            checked_args = []
            for count, (type, arg) in enumerate(zip(types, args)):
                if type is None:
                    checked_args.append(arg)
                    continue

                if type is not arg.type:
                    raise TypeError(f"Positional argument {count} in procedure {func.__name__} requires a {type.name}, but a {arg.type.name} was given")

                checked_args.append(arg.value)

            return func(*checked_args)
        return ExternalProcedure.new(func.__name__, inner)
    return wrapped

@auto_external(None)
def DISPLAY(value):
    def stringify(value):
        if value.type is MemType.BOOLEAN:
            return "TRUE" if value.value else "FALSE"

        if value.type is not MemType.LIST:
            return str(value.value)

        return "[" + ", ".join([stringify(i) for i in value.value]) + "]"

    out = stringify(value).replace("\\n", "\n")
    print(out, end=" ")

def index_oob_error(index, list):
    if index > len(list) or index <= 0:
        raise IndexError(f"Index {index} out of bounds")

@auto_external()
def INPUT():
    return MemEntry(input(), MemType.STRING)

@auto_external(MemType.LIST, MemType.INTEGER, None)
def INSERT(list: list, index: int, item):
    if index > len(list) + 1 or index <= 0:
        raise IndexError(f"Index shoot me {index} out of bounds")
    list.insert(index - 1, item)

@auto_external(MemType.LIST, None)
def APPEND(list: list, item):
    list.append(item)

@auto_external(MemType.LIST, MemType.INTEGER)
def REMOVE(list: list, index: int):
    if index > len(list) or index <= 0:
        raise IndexError(f"Index {index} out of bounds")
    del list[index - 1]

@auto_external(MemType.LIST)
def LENGTH(list: list):
    return MemEntry(len(list), MemType.INTEGER)

@auto_external(MemType.INTEGER, MemType.INTEGER)
def RANDOM(a: int, b: int):
    return MemEntry(randint(a, b), MemType.INTEGER)

@auto_external(MemType.INTEGER, MemType.INTEGER)
def SETUP(x, y):
    Robot().setup(x, y)

@auto_external(MemType.INTEGER, MemType.INTEGER)
def CLOSE_BOX(x, y):
    Robot().box(x-1, y-1)

@auto_external(MemType.LIST, MemType.LIST)
def CLOSE_MULTI_BOX(x_ls: list, y_ls: list):
    def mem_unwrapper(value: MemEntry):
        if value.type is MemType.INTEGER:
            return value.value
        raise TypeError(f"CLOSE_MULTI_BOX requires a list of INTEGER, not {value.type}")

    x_ls = list(map(mem_unwrapper, x_ls))
    y_ls = list(map(mem_unwrapper, y_ls))

    for x, y in zip(x_ls, y_ls):
        Robot().box(x-1, y-1)

@auto_external()
def MOVE_FORWARD():
    Robot().forward()

@auto_external()
def ROTATE_LEFT():
    Robot().rotate_left()

@auto_external()
def ROTATE_RIGHT():
    Robot().rotate_right()

@auto_external(MemType.STRING)
def CAN_MOVE(direction):
    return MemEntry(Robot().can_move(direction), MemType.BOOLEAN)

PRELUDE = {
    external.value.name: external for external in [
        INPUT,
        DISPLAY,
        INSERT,
        APPEND,
        REMOVE,
        LENGTH,
        RANDOM,
        SETUP,
        CLOSE_BOX,
        CLOSE_MULTI_BOX,
        MOVE_FORWARD,
        ROTATE_LEFT,
        ROTATE_RIGHT,
        CAN_MOVE
]}


class NodeVisiter:
    def visit(self, node: Node):
        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self,node):
        raise Exception('No visit_{} method'.format(type(node).__name__))


class Interpreter(NodeVisiter):
    def __init__(self, parser):
        self.parser = parser
        self.stack = CallStack()
        self.stack.push(PRELUDE)

        # flags
        self.logging = False

    def log(self, *out, sep=", "):
        if self.logging:
            print(OKCYAN, end="")
            print(*out, sep=sep, end="")
            print(ENDC, end="\n")

    def visit(self, node: Node):
        value = super().visit(node)
        if value is None:
            return MemEntry(None, MemType.NONE)
        return value

    def interpret(self):
        self.log("INITIAL STACK")
        self.log(self.stack)

        tree = self.parser.parse()
        out = self.visit(tree)

        self.log("\nFINAL STACK")
        self.log(self.stack)

        return out

    ## node visiters ##

    def visit_Block(self, node: Block):
        for child in node.children:
            val = self.visit(child)
            if val.type is MemType.RETURN:
                return val

        return MemEntry(None, MemType.NONE)

    def visit_Return(self, node: Return):
        value = self.visit(node.value)

        self.log(f"Returning {value}")

        return MemEntry(value, MemType.RETURN)

    def visit_Assign(self, node: Assign):
        access = node.var
        value = self.visit(node.value)
        if value.type is MemType.LIST:
            value = MemEntry(copy(value.value), MemType.LIST)

        if type(access) is ListAccess:
            index = self.visit(access.index)
            if index.type is not MemType.INTEGER:
                raise TypeError(f"Cannot index list with type: {index.type.name}, must use {MemType.INTEGER.name}")
            index = index.value - 1

            _list = self.visit(access.list)
            if _list.type is not MemType.LIST:
                raise TypeError(f"Cannot index {_list.type.name}")
            _list = _list.value

            _list[index] = value

        elif type(access) is Var:
            name = node.var.value
            self.stack[name] = value

            self.log(f"\nNEW ASSIGNMENT FOR {name}")
            self.log(self.stack)

    def visit_Procedure(self, node: Procedure):
        return MemEntry(node, MemType.PROCEDURE)

    def visit_IfStatement(self, node: IfStatement):
        condition = self.visit(node.condition)
        if condition.type is not MemType.BOOLEAN:
            raise TypeError(f"If statement requires boolean condition, but {condition.type} was found")

        if condition.value:
            return self.visit(node.true_block)

        if node.else_block is not None:
            return self.visit(node.else_block)

    def visit_IncrementLoop(self, node: IncrementLoop):
        value = self.visit(node.value)
        if value.type is not MemType.INTEGER:
            raise TypeError(f"Cannot increment to type: {value.type.name}, must be {MemType.INTEGER.name}")

        for _ in range(value.value):
            out = self.visit(node.block)
            if out.type is MemType.RETURN:
                return out

    def visit_UntilLoop(self, node: UntilLoop):
        while True:
            condition = self.visit(node.condition)
            if condition.type is not MemType.BOOLEAN:
                raise TypeError(f"Condition in REPEAT UNTIL block must be {MemType.BOOLEAN.name}, not {condition.type.name}")

            if condition.value:
                break

            out = self.visit(node.block)
            if out.type is MemType.RETURN:
                return out

    def visit_ForEachLoop(self, node: ForEachLoop):
        name = node.id.value

        _list = self.visit(node.list)
        if _list.type is not MemType.LIST:
            raise TypeError(f"FOR EACH loop can only increment over {MemType.LIST.name}. Cannot increment over {_list.type.name}")

        for item in _list.value:
            self.stack[name] = item
            out = self.visit(node.block)
            if out.type is MemType.RETURN:
                return out

    def visit_NoOp(self, node: NoOp):
        pass

    def visit_BinOp(self, node: BinOp):
        left = self.visit(node.left)
        right = self.visit(node.right)

        return BINARY_OPERATORS[node.oper.type](left, right)

    def visit_UnaryOp(self, node: UnaryOp):
        self.log(f"Visiting Unary Operator")

        arg = self.visit(node.expr)
        self.log(f"Argument: {arg}")
        self.log(f"Operator: {node.oper.type}")

        return UNARY_OPERATORS[node.oper.type](arg)

    def visit_Primative(self, node: Primative):
        return MemEntry(node.value, PRIMATIVE_TO_MEMTYPE[node.token.type])

    def visit_List(self, node: List):
        items = list(map(self.visit, node.items))
        return MemEntry(items, MemType.LIST)

    def visit_ListAccess(self, node: ListAccess):
        index = self.visit(node.index)
        if index.type is not MemType.INTEGER:
            raise TypeError(f"Cannot index list with type: {index.type.name}, must use {MemType.INTEGER.name}")
        index = index.value - 1

        _list = self.visit(node.list)
        if _list.type is not MemType.LIST:
            raise TypeError(f"Cannot index {_list.type.name}")
        _list = _list.value

        return _list[index]

    def visit_Var(self, node: Var):
        name = node.value
        val = self.stack[name]

        self.log(f"\nNEW ACCESS FOR {name}")
        self.log(self.stack)

        if val is None:
            raise NameError(repr(name))
        else:
            return val

    def visit_Call(self, node: Call):
        procedure = self.visit(node.var)
        literal_args = [self.visit(a) for a in node.args]

        if procedure.type is MemType.EXTERNAL_PROCEDURE:
            return procedure.value.func(literal_args)
        
        elif procedure.type is MemType.PROCEDURE:
            formal_args = [a.value for a in procedure.value.args]

            if len(formal_args) != len(literal_args):
                raise SyntaxError(f"{node.var.value} expects {len(formal_args)} argument(s) but {len(literal_args)} were supplied")

            self.stack.push(dict(zip(formal_args, literal_args)))

            self.log("\nPROCEDURE CALL")
            self.log(f"Running {procedure.value}")
            self.log(self.stack)

            val = self.visit(procedure.value.block)   
            self.stack.pop()

            if val.type is MemType.RETURN:
                return val.value

        else:
            raise TypeError(f"Type {procedure.type} is not callable")

def main():
    with open("code.apcsp") as file:
        text = file.read()

    print(text)
    print("\n----------------------------------------\n")
    
    lexer = Lexer(text)
    parser = Parser(lexer)
    interpreter = Interpreter(parser)
    interpreter.interpret()
    if Robot.instance is not None:
        tk.mainloop()

if __name__ == '__main__':
    main()
