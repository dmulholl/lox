#! /usr/bin/env python3
# ------------------------------------------------------------------------------
# Lox implemented in Python 3.
# ------------------------------------------------------------------------------

import sys
import enum


# ------------------------------------------------------------------------------
# Lexer.
# ------------------------------------------------------------------------------


def is_digit(char):
    return ord(char) >= ord('0') and ord(char) <= ord('9')


def is_alpha(char):
    if ord(char) >= ord('a') and ord(char) <= ord('z'):
        return True
    if ord(char) >= ord('A') and ord(char) <= ord('Z'):
        return True
    if char == '_':
        return True
    return False


def is_alphanumeric(char):
    return is_digit(char) or is_alpha(char)


class TokType(enum.Enum):
    LeftParen = enum.auto()
    RightParen = enum.auto()
    LeftBrace = enum.auto()
    RightBrace = enum.auto()
    Comma = enum.auto()
    Dot = enum.auto()
    Minus = enum.auto()
    Plus = enum.auto()
    Semicolon = enum.auto()
    Slash = enum.auto()
    Star = enum.auto()
    Bang = enum.auto()
    BangEqual = enum.auto()
    Equal = enum.auto()
    EqualEqual = enum.auto()
    Greater = enum.auto()
    GreaterEqual = enum.auto()
    Less = enum.auto()
    LessEqual = enum.auto()
    Identifier = enum.auto()
    String = enum.auto()
    Number = enum.auto()
    And = enum.auto()
    Class = enum.auto()
    Else = enum.auto()
    TokFalse = enum.auto()
    Fun = enum.auto()
    For = enum.auto()
    If = enum.auto()
    Nil = enum.auto()
    Or = enum.auto()
    Print = enum.auto()
    Return = enum.auto()
    Super = enum.auto()
    This = enum.auto()
    TokTrue = enum.auto()
    Var = enum.auto()
    While = enum.auto()
    Eof = enum.auto()


keywords = {
    'and': TokType.And,
    'class': TokType.Class,
    'else': TokType.Else,
    'false': TokType.TokFalse,
    'for': TokType.For,
    'fun': TokType.Fun,
    'if': TokType.If,
    'nil': TokType.Nil,
    'or': TokType.Or,
    'print': TokType.Print,
    'return': TokType.Return,
    'super': TokType.Super,
    'this': TokType.This,
    'true': TokType.TokTrue,
    'var': TokType.Var,
    'while': TokType.While,
}


class Token:

    def __init__(self, type: TokType, lexeme: str, literal, line: int):
        self.type = type
        self.lexeme = lexeme
        self.literal = literal
        self.line = line

    def __repr__(self):
        return f"{self.type} \"{self.lexeme}\" <{self.literal}>"


class Scanner:

    def __init__(self, source: str):
        self.source = source
        self.tokens = list()
        self.start = 0
        self.current = 0
        self.line = 1

    def scan_tokens(self):
        while not self.is_at_end():
            self.start = self.current
            self.scan_token()
        self.tokens.append(Token(TokType.Eof, "", None, self.line))
        return self.tokens

    def is_at_end(self):
        return self.current >= len(self.source)

    def scan_token(self):
        c = self.advance()

        # Single character tokens.
        if c == '(':
            self.add_token(TokType.LeftParen)
        elif c == ')':
            self.add_token(TokType.RightParen)
        elif c == '{':
            self.add_token(TokType.LeftBrace)
        elif c == '}':
            self.add_token(TokType.RightBrace)
        elif c == ',':
            self.add_token(TokType.Comma)
        elif c == '.':
            self.add_token(TokType.Dot)
        elif c == '-':
            self.add_token(TokType.Minus)
        elif c == '+':
            self.add_token(TokType.Plus)
        elif c == ';':
            self.add_token(TokType.Semicolon)
        elif c == '*':
            self.add_token(TokType.Star)

        # Double or single character tokens.
        elif c == '!':
            if self.match('='):
                self.add_token(TokType.BangEqual)
            else:
                self.add_token(TokType.Bang)
        elif c == '=':
            if self.match('='):
                self.add_token(TokType.EqualEqual)
            else:
                self.add_token(TokType.Equal)
        elif c == '<':
            if self.match('='):
                self.add_token(TokType.LessEqual)
            else:
                self.add_token(TokType.Equal)
        elif c == '>':
            if self.match('='):
                self.add_token(TokType.GreaterEqual)
            else:
                self.add_token(TokType.Equal)

        # Slash or comment.
        elif c == '/':
            if self.match('/'):
                while self.peek() != '\n' and not self.is_at_end():
                    self.advance()
            else:
                self.add_token(TokType.Slash)

        # Whitespace.
        elif c == ' ' or c == '\r' or c == '\t':
            pass
        elif c == '\n':
            self.line += 1

        # Strings.
        elif c == '"':
            self.read_string()

        # Numbers.
        elif is_digit(c):
            self.read_number()

        # Identifiers & keywords.
        elif is_alpha(c):
            self.read_identifier()

        else:
            lexing_error(self.line, f"Unexpected character '{c}'.")

    def advance(self):
        self.current += 1
        return self.source[self.current - 1]

    def match(self, char):
        if self.is_at_end():
            return False
        if self.source[self.current] != char:
            return False;
        self.current += 1
        return True

    def peek(self):
        if self.is_at_end():
            return '\0'
        return self.source[self.current]

    def peek_next(self):
        if self.current + 1 >= len(self.source):
            return '\0'
        return self.source[self.current + 1]

    def add_token(self, type, literal=None):
        text = self.source[self.start:self.current]
        self.tokens.append(Token(type, text, literal, self.line))

    def read_string(self):
        while self.peek() != '"' and not self.is_at_end():
            if self.peek() == '\n':
                self.line += 1
            self.advance()
        if self.is_at_end():
            lexing_error(self.line, "Unterminated string.")
            return
        self.advance()
        value = self.source[self.start + 1:self.current - 1]
        self.add_token(TokType.String, value)

    def read_number(self):
        while self.peek().isdigit():
            self.advance()
        if self.peek() == '.' and self.peek_next().isdigit():
            self.advance()
            while self.peek().isdigit():
                self.advance()
        value = float(self.source[self.start:self.current])
        self.add_token(TokType.Number, value)

    def read_identifier(self):
        while is_alphanumeric(self.peek()):
            self.advance()
        text = self.source[self.start:self.current]
        if text in keywords:
            self.add_token(keywords[text])
        else:
            self.add_token(TokType.Identifier)


# ------------------------------------------------------------------------------
# Expressions.
# ------------------------------------------------------------------------------


class Expr:
    pass


class BinaryExpr(Expr):
    def __init__(self, left: Expr, operator: Token, right: Expr):
        self.left = left
        self.operator = operator
        self.right = right


class GroupingExpr(Expr):
    def __init__(self, expression: Expr):
        self.expression = expression


class LiteralExpr(Expr):
    def __init__(self, value):
        self.value = value


class UnaryExpr(Expr):
    def __init__(self, operator: Token, right: Expr):
        self.operator = operator
        self.right = right


# ------------------------------------------------------------------------------
# Parser.
# ------------------------------------------------------------------------------


class ParsingError(Exception):
    pass


class Parser:

    def __init__(self, tokens):
        self.tokens = tokens
        self.current = 0

    def parse(self):
        try:
            return self.expression()
        except ParsingError:
            return None

    # ------------------------------------------------------------------------
    # Expression parsers.
    # ------------------------------------------------------------------------

    def expression(self):
        return self.equality()

    def equality(self):
        expr = self.comparison()
        while self.match(TokType.BangEqual, TokType.EqualEqual):
            operator = self.previous()
            right = self.comparison()
            expr = BinaryExpr(expr, operator, right)
        return expr

    def comparison(self):
        expr = self.addition()
        operators = (
            TokType.Greater,
            TokType.GreaterEqual,
            TokType.Less,
            TokType.LessEqual
        )
        while self.match(*operators):
            operator = self.previous()
            right = self.addition()
            expr = BinaryExpr(expr, operator, right)
        return expr

    def addition(self):
        expr = self.multiplication()
        while self.match(TokType.Minus, TokType.Plus):
            operator = self.previous()
            right = self.multiplication()
            expr = BinaryExpr(expr, operator, right)
        return expr

    def multiplication(self):
        expr = self.unary()
        while self.match(TokType.Slash, TokType.Star):
            operator = self.previous()
            right = self.unary()
            expr = BinaryExpr(expr, operator, right)
        return expr

    def unary(self):
        if self.match(TokType.Bang, TokType.Minus):
            operator = self.previous()
            right = self.unary()
            return UnaryExpr(operator, right)
        return self.primary()

    def primary(self):
        if self.match(TokType.TokFalse):
            return LiteralExpr(False)
        if self.match(TokType.TokTrue):
            return LiteralExpr(True)
        if self.match(TokType.Nil):
            return LiteralExpr(None)
        if self.match(TokType.Number, TokType.String):
            return LiteralExpr(self.previous().literal)
        if self.match(TokType.LeftParen):
            expr = self.expression()
            self.consume(TokType.RightParen, "Expect ')' after expression.")
            return GroupingExpr(expr)
        parsing_error(self.peek(), "Expect expression.")
        raise ParsingError()

    # ------------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------------

    def match(self, *toktypes):
        for toktype in toktypes:
            if self.check(toktype):
                self.advance()
                return True
        return False

    def check(self, toktype):
        if self.is_at_end():
            return False
        return self.peek().type == toktype

    def advance(self):
        if not self.is_at_end():
            self.current += 1
        return self.previous()

    def is_at_end(self):
        return self.peek().type == TokType.Eof

    def peek(self):
        return self.tokens[self.current]

    def previous(self):
        return self.tokens[self.current - 1]

    def consume(self, toktype, message):
        if self.check(toktype):
            return self.advance()
        parsing_error(self.peek(), message)
        raise ParsingError()

    def synchronize(self):
        self.advance()
        while not self.is_at_end():
            if self.previous().type == TokType.Semicolon:
                return
            if self.peek().type in (
                TokType.Class,
                TokType.Fun,
                TokType.Var,
                TokType.For,
                TokType.If,
                TokType.While,
                TokType.Print,
                TokType.Return,
            ):
                return
            self.advance()


# ------------------------------------------------------------------------------
# Printer.
# ------------------------------------------------------------------------------


class Printer:

    def tostring(self, expr: Expr):
        if isinstance(expr, BinaryExpr):
            return self.parenthesize(expr.operator.lexeme, expr.left, expr.right)
        elif isinstance(expr, GroupingExpr):
            return self.parenthesize("group", expr.expression)
        elif isinstance(expr, LiteralExpr):
            if expr.value == None:
                return "nil"
            else:
                return str(expr.value)
        elif isinstance(expr, UnaryExpr):
            return self.parenthesize(expr.operator.lexeme, expr.right)

    def parenthesize(self, name: str, *exprs):
        output = f"({name}"
        for expr in exprs:
            output += " "
            output += self.tostring(expr)
        output += ")"
        return output


# ------------------------------------------------------------------------------
# Interpreter.
# ------------------------------------------------------------------------------


class RuntimeError(Exception):
    def __init__(self, token: Token, message: str):
        self.token = token
        self.message = message


class Interpreter:

    def __init__(self):
        pass

    def interpret(self, expr):
        try:
            value = self.eval(expr)
            print(self.stringify(value))
        except RuntimeError as error:
            runtime_error(error)

    def eval(self, expr: Expr):
        if isinstance(expr, LiteralExpr):
            return self.eval_literal(expr)
        elif isinstance(expr, GroupingExpr):
            return self.eval_grouping(expr)
        elif isinstance(expr, UnaryExpr):
            return self.eval_unary(expr)
        elif isinstance(expr, BinaryExpr):
            return self.eval_binary(expr)

    def eval_literal(self, expr):
        return expr.value

    def eval_grouping(self, expr):
        return self.eval(expr.expression)

    def eval_unary(self, expr):
        right = self.eval(expr.right)
        if expr.operator.type == TokType.Minus:
            self.check_float_operand(expr.operator, right)
            return -right
        elif expr.operator.type == TokType.Bang:
            return not self.is_truthy(right)

    def eval_binary(self, expr):
        left = self.eval(expr.left)
        right = self.eval(expr.right)

        if expr.operator.type == TokType.Greater:
            self.check_float_operands(expr.operator, left, right)
            return left > right
        elif expr.operator.type == TokType.GreaterEqual:
            self.check_float_operands(expr.operator, left, right)
            return left >= right
        elif expr.operator.type == TokType.Less:
            self.check_float_operands(expr.operator, left, right)
            return left < right
        elif expr.operator.type == TokType.LessEqual:
            self.check_float_operands(expr.operator, left, right)
            return left <= right
        elif expr.operator.type == TokType.Minus:
            self.check_float_operands(expr.operator, left, right)
            return left - right
        elif expr.operator.type == TokType.Plus:
            self.check_float_or_str_operands(expr.operator, left, right)
            return left + right
        elif expr.operator.type == TokType.Slash:
            self.check_float_operands(expr.operator, left, right)
            return left / right
        elif expr.operator.type == TokType.Star:
            self.check_float_operands(expr.operator, left, right)
            return left * right
        elif expr.operator.type == TokType.BangEqual:
            return not self.is_equal(left, right)
        elif expr.operator.type == TokType.EqualEqual:
            return self.is_equal(left, right)


    def is_truthy(self, value):
        if value is None:
            return False
        if value is False:
            return False
        return True

    def is_equal(self, val_a, val_b):
        return val_a == val_b

    def check_float_operand(self, operator: Token, operand):
        if isinstance(operand, float):
            return
        raise RuntimeError(operator, "Operand must be a number.")

    def check_float_operands(self, operator: Token, left, right):
        if isinstance(left, float) and isinstance(right, float):
            return
        raise RuntimeError(operator, "Operands must be numbers.")

    def check_float_or_str_operands(self, operator: Token, left, right):
        if isinstance(left, float) and isinstance(right, float):
            return
        if isinstance(left, str) and isinstance(right, str):
            return
        raise RuntimeError(
            operator, "Operands must be two numbers or two strings."
        )

    def stringify(self, value):
        if value is None:
            return "nil"
        if value is True:
            return "true"
        if value is False:
            return "false"
        if isinstance(value, float):
            text = str(value)
            if text.endswith('.0'):
                text = text[:-2]
            return text
        return str(value)



# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


had_lexing_error = False
had_parsing_error = False
had_runtime_error = False
interpreter = Interpreter()


def lexing_error(line: int, message: str):
    print(f"[line {line}] Lexing Error: {message}")
    global had_lexing_error
    had_lexing_error = True


def parsing_error(tok: Token, message: str):
    if tok.type == TokType.Eof:
        print(f"[line {tok.line}] Parsing Error at end: {message}")
    else:
        print(f"[line {tok.line}] Parsing Error at '{tok.lexeme}': {message}")
    global had_parsing_error
    had_parsing_error = True


def runtime_error(error: RuntimeError):
    print(f"[line {error.token.line}] Runtime Error: {error.message}")
    global had_runtime_error
    had_runtime_error = True


def run(source):

    global had_lexing_error, had_parsing_error, had_runtime_error
    had_lexing_error = False
    had_parsing_error = False
    had_runtime_error = False

    scanner = Scanner(source)
    tokens = scanner.scan_tokens()
    #print(tokens)
    if had_lexing_error:
        return

    parser = Parser(tokens)
    expr = parser.parse()
    #print(Printer().tostring(expr))
    if had_parsing_error:
        return

    interpreter.interpret(expr)


def run_prompt():
    try:
        while True:
            print("> ", end="")
            run(input())
    except KeyboardInterrupt:
        print()


def run_file(path):
    run(open(path).read())
    if had_error:
        sys.exit(1)
    if had_runtime_error:
        sys.exit(2)


def main():
    if len(sys.argv) > 2:
        print("Usage: lox [script]")
    elif len(sys.argv) == 2:
        run_file(sys.argv[1])
    else:
        run_prompt()


if __name__ == '__main__':
    main()
