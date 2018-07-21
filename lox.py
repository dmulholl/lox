#! /usr/bin/env python3
# ------------------------------------------------------------------------------
# Bob Nystrom's Lox language implemented in Python.
# ------------------------------------------------------------------------------

import sys
import enum
import time

from typing import List, Any, Union, Dict


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
                self.add_token(TokType.Less)
        elif c == '>':
            if self.match('='):
                self.add_token(TokType.GreaterEqual)
            else:
                self.add_token(TokType.Greater)

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


class VariableExpr(Expr):
    def __init__(self, name: Token):
        self.name = name


class AssignExpr(Expr):
    def __init__(self, name: Token, value: Expr):
        self.name = name
        self.value = value


class LogicalExpr(Expr):
    def __init__(self, left: Expr, operator: Token, right: Expr):
        self.left = left
        self.operator = operator
        self.right = right


class CallExpr(Expr):
    def __init__(self, callee: Expr, paren: Token, arguments: List[Expr]):
        self.callee = callee
        self.paren = paren
        self.arguments = arguments


class GetExpr(Expr):
    def __init__(self, object: Expr, name: Token):
        self.object = object
        self.name = name


class SetExpr(Expr):
    def __init__(self, object: Expr, name: Token, value: Expr):
        self.object = object
        self.name = name
        self.value = value


class ThisExpr(Expr):
    def __init__(self, keyword: Token):
        self.keyword = keyword


class SuperExpr(Expr):
    def __init__(self, keyword: Token, method: Token):
        self.keyword = keyword
        self.method = method


# ------------------------------------------------------------------------------
# Statements.
# ------------------------------------------------------------------------------


class Stmt:
    pass


class ExpressionStmt(Stmt):
    def __init__(self, expression: Expr):
        self.expression = expression


class PrintStmt(Stmt):
    def __init__(self, expression: Expr):
        self.expression = expression


class VarStmt(Stmt):
    def __init__(self, name: Token, initializer: Expr):
        self.name = name
        self.initializer = initializer


class BlockStmt(Stmt):
    def __init__(self, statements: List[Stmt]):
        self.statements = statements


class IfStmt(Stmt):
    def __init__(self, condition: Expr, then_branch: Stmt, else_branch: Stmt):
        self.condition = condition
        self.then_branch = then_branch
        self.else_branch = else_branch


class WhileStmt(Stmt):
    def __init__(self, condition: Expr, body: Stmt):
        self.condition = condition
        self.body = body


class FunctionStmt(Stmt):
    def __init__(self, name: Token, parameters: List[Token], body: List[Stmt]):
        self.name = name
        self.parameters = parameters
        self.body = body


class ReturnStmt(Stmt):
    def __init__(self, keyword: Token, value: Expr):
        self.keyword = keyword
        self.value = value


class ClassStmt(Stmt):
    def __init__(self, name: Token, superclass: VariableExpr, methods: List[FunctionStmt]):
        self.name = name
        self.methods = methods
        self.superclass = superclass


# ------------------------------------------------------------------------------
# Parser.
# ------------------------------------------------------------------------------


class ParsingError(Exception):
    pass


class Parser:

    def __init__(self, tokens):
        self.tokens = tokens
        self.current = 0

    def parse(self) -> List[Stmt]:
        statements = list()
        while not self.is_at_end():
            statements.append(self.declaration())
        return statements

    # ------------------------------------------------------------------------
    # Statement parsers.
    # ------------------------------------------------------------------------

    def declaration(self):
        try:
            if self.match(TokType.Var):
                return self.var_statement()
            if self.match(TokType.Fun):
                return self.function("function")
            if self.match(TokType.Class):
                return self.class_statement()
            return self.statement()
        except ParsingError:
            self.synchronize()

    def var_statement(self):
        name = self.consume(TokType.Identifier, "Expect variable name.")
        initializer = None
        if self.match(TokType.Equal):
            initializer = self.expression()
        self.consume(TokType.Semicolon, "Expect ';' after variable declaration.")
        return VarStmt(name, initializer)

    def statement(self):
        if self.match(TokType.Print):
            return self.print_statement()
        elif self.match(TokType.If):
            return self.if_statement()
        elif self.match(TokType.While):
            return self.while_statement()
        elif self.match(TokType.For):
            return self.for_statement()
        elif self.match(TokType.LeftBrace):
            return BlockStmt(self.block_statement())
        elif self.match(TokType.Return):
            return self.return_statement()
        return self.expression_statement()

    def print_statement(self):
        expr = self.expression()
        self.consume(TokType.Semicolon, "Expect ';' after value.")
        return PrintStmt(expr)

    def expression_statement(self):
        expr = self.expression()
        self.consume(TokType.Semicolon, "Expect ';' after expression.")
        return ExpressionStmt(expr)

    def block_statement(self):
        statements = []
        while not self.check(TokType.RightBrace) and not self.is_at_end():
            statements.append(self.declaration())
        self.consume(TokType.RightBrace, "Expect '}' after block.")
        return statements

    def if_statement(self):
        self.consume(TokType.LeftParen, "Expect '(' after 'if'.");
        condition = self.expression()
        self.consume(TokType.RightParen, "Expect ')' after if condition.")
        then_branch = self.statement()
        else_branch = None
        if self.match(TokType.Else):
            else_branch = self.statement()
        return IfStmt(condition, then_branch, else_branch)

    def while_statement(self):
        self.consume(TokType.LeftParen, "Expect '(' after 'while'.");
        condition = self.expression()
        self.consume(TokType.RightParen, "Expect ')' after condition.")
        body = self.statement()
        return WhileStmt(condition, body)

    def for_statement(self):
        self.consume(TokType.LeftParen, "Expect '(' after 'for'.");
        initializer, condition, increment = None, None, None

        if self.match(TokType.Semicolon):
            initializer = None
        elif self.match(TokType.Var):
            initializer = self.var_statement()
        else:
            initializer = self.expression_statement()

        if not self.check(TokType.Semicolon):
            condition = self.expression()
        self.consume(TokType.Semicolon, "Expect ';' after loop condition.")

        if not self.check(TokType.RightParen):
            increment = self.expression()
        self.consume(TokType.RightParen, "Expect ')' after 'for' clauses.")

        body = self.statement()
        if increment is not None:
            body = BlockStmt([body, ExpressionStmt(increment)])
        if condition is None:
            condition = LiteralExpr(True)
        body = WhileStmt(condition, body)
        if initializer is not None:
            body = BlockStmt([initializer, body])
        return body

    def function(self, kind: str):
        name = self.consume(TokType.Identifier, f"Expect {kind} name.")
        self.consume(TokType.LeftParen, f"Expect '(' after {kind} name.")
        parameters = list()
        if not self.check(TokType.RightParen):
            while True:
                if len(parameters) >= 8:
                    parsing_error(self.peek(), "Cannot have more than 8 parameters.")
                parameters.append(self.consume(TokType.Identifier, "Expect parameter name."))
                if not self.match(TokType.Comma):
                    break
        self.consume(TokType.RightParen, "Expect ')' after parameters.")
        self.consume(TokType.LeftBrace, f"Expect '{{' before {kind} body.")
        body = self.block_statement()
        return FunctionStmt(name, parameters, body)

    def return_statement(self):
        keyword = self.previous()
        value = None
        if not self.check(TokType.Semicolon):
            value = self.expression()
        self.consume(TokType.Semicolon, "Expect ';' after return value.")
        return ReturnStmt(keyword, value)

    def class_statement(self):
        name = self.consume(TokType.Identifier, "Expect class name.")
        superclass = None
        if self.match(TokType.Less):
            self.consume(TokType.Identifier, "Expect superclass name.")
            superclass = VariableExpr(self.previous())
        self.consume(TokType.LeftBrace, "Expect '{' before class body.")
        methods = list()
        while not self.check(TokType.RightBrace) and not self.is_at_end():
            methods.append(self.function("method"))
        self.consume(TokType.RightBrace, "Expect '}' after class body.")
        return ClassStmt(name, superclass, methods)

    # ------------------------------------------------------------------------
    # Expression parsers.
    # ------------------------------------------------------------------------

    def expression(self):
        return self.assignment()

    def assignment(self):
        expr = self.logical_or()
        if self.match(TokType.Equal):
            equals = self.previous()
            value = self.assignment()
            if isinstance(expr, VariableExpr):
                return AssignExpr(expr.name, value)
            elif isinstance(expr, GetExpr):
                return SetExpr(expr.object, expr.name, value)
            parsing_error(equals, "Invalid assignment target.")
        return expr

    def logical_or(self):
        expr = self.logical_and()
        while self.match(TokType.Or):
            operator = self.previous()
            right = self.logical_and()
            expr = LogicalExpr(expr, operator, right)
        return expr

    def logical_and(self):
        expr = self.equality()
        while self.match(TokType.And):
            operator = self.previous()
            right = self.equality()
            expr = LogicalExpr(expr, operator, right)
        return expr

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
        return self.call()

    def call(self):
        expr = self.primary()
        while True:
            if self.match(TokType.LeftParen):
                expr = self.finish_call(expr)
            elif self.match(TokType.Dot):
                name = self.consume(
                    TokType.Identifier,
                    "Expect property name after '.'."
                )
                expr = GetExpr(expr, name)
            else:
                break
        return expr

    def finish_call(self, callee: Expr):
        arguments = list()
        if not self.check(TokType.RightParen):
            while True:
                if len(arguments) >= 8:
                    parsing_error(self.peek(), "Cannot have more than 8 arguments.")
                arguments.append(self.expression())
                if not self.match(TokType.Comma):
                    break
        paren = self.consume(TokType.RightParen, "Expect ')' after arguments.")
        return CallExpr(callee, paren, arguments)

    def primary(self):
        if self.match(TokType.TokFalse):
            return LiteralExpr(False)
        if self.match(TokType.TokTrue):
            return LiteralExpr(True)
        if self.match(TokType.Nil):
            return LiteralExpr(None)
        if self.match(TokType.Super):
            keyword = self.previous()
            self.consume(TokType.Dot, "Expect '.' after 'super'.")
            method = self.consume(
                TokType.Identifier,
                "Expect superclass method name."
            )
            return SuperExpr(keyword, method)
        if self.match(TokType.This):
            return ThisExpr(self.previous())
        if self.match(TokType.Number, TokType.String):
            return LiteralExpr(self.previous().literal)
        if self.match(TokType.LeftParen):
            expr = self.expression()
            self.consume(TokType.RightParen, "Expect ')' after expression.")
            return GroupingExpr(expr)
        if self.match(TokType.Identifier):
            return VariableExpr(self.previous())
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
# Variable resolver.
# ------------------------------------------------------------------------------


class FunctionType(enum.Enum):
    NoFunc = enum.auto()
    Function = enum.auto()
    Method = enum.auto()
    Initializer = enum.auto()


class ClassType(enum.Enum):
    NoClass = enum.auto()
    Class = enum.auto()
    Subclass = enum.auto()


class Resolver:

    def __init__(self, interpreter: 'Interpreter'):
        self.interpreter = interpreter
        self.scopes = list()
        self.current_function = FunctionType.NoFunc
        self.current_class = ClassType.NoClass

    def resolve_program(self, program: List[Stmt]):
        for statement in program:
            self.resolve(statement)

    def begin_scope(self):
        self.scopes.append(dict())

    def end_scope(self):
        self.scopes.pop()

    def resolve(self, item: Union[Stmt, Expr]):
        if isinstance(item, BlockStmt):
            self.resolve_block_stmt(item)
        elif isinstance(item, VarStmt):
            self.resolve_var_stmt(item)
        elif isinstance(item, VariableExpr):
            self.resolve_var_expr(item)
        elif isinstance(item, AssignExpr):
            self.resolve_assign_expr(item)
        elif isinstance(item, FunctionStmt):
            self.resolve_function_stmt(item)
        elif isinstance(item, ExpressionStmt):
            self.resolve_expression_stmt(item)
        elif isinstance(item, IfStmt):
            self.resolve_if_stmt(item)
        elif isinstance(item, PrintStmt):
            self.resolve_print_stmt(item)
        elif isinstance(item, ReturnStmt):
            self.resolve_return_stmt(item)
        elif isinstance(item, WhileStmt):
            self.resolve_while_stmt(item)
        elif isinstance(item, BinaryExpr):
            self.resolve_binary_expr(item)
        elif isinstance(item, CallExpr):
            self.resolve_call_expr(item)
        elif isinstance(item, GroupingExpr):
            self.resolve_grouping_expr(item)
        elif isinstance(item, LiteralExpr):
            self.resolve_literal_expr(item)
        elif isinstance(item, LogicalExpr):
            self.resolve_logical_expr(item)
        elif isinstance(item, UnaryExpr):
            self.resolve_unary_expr(item)
        elif isinstance(item, ClassStmt):
            self.resolve_class_stmt(item)
        elif isinstance(item, GetExpr):
            self.resolve_get_expr(item)
        elif isinstance(item, SetExpr):
            self.resolve_set_expr(item)
        elif isinstance(item, ThisExpr):
            self.resolve_this_expr(item)
        elif isinstance(item, SuperExpr):
            self.resolve_super_expr(item)
        else:
            print('resolver error!')

    def resolve_list(self, statements: List[Stmt]):
        for statement in statements:
            self.resolve(statement)

    def resolve_block_stmt(self, stmt: BlockStmt):
        self.begin_scope()
        self.resolve_list(stmt.statements)
        self.end_scope()

    def resolve_var_stmt(self, stmt: VarStmt):
        self.declare(stmt.name)
        if stmt.initializer is not None:
            self.resolve(stmt.initializer)
        self.define(stmt.name)

    def resolve_var_expr(self, expr: VariableExpr):
        if self.scopes:
            if expr.name.lexeme in self.scopes[-1]:
                if self.scopes[-1][expr.name.lexeme] == False:
                    parsing_error(
                        expr.name,
                        "Cannot read local variable in its own initializer."
                    )
        self.resolve_local(expr, expr.name)

    def resolve_assign_expr(self, expr: AssignExpr):
        self.resolve(expr.value)
        self.resolve_local(expr, expr.name)

    def resolve_function_stmt(self, stmt: FunctionStmt):
        self.declare(stmt.name)
        self.define(stmt.name)
        self.resolve_function(stmt, FunctionType.Function)

    def resolve_function(self, stmt: FunctionStmt, type: FunctionType):
        enclosing_function = self.current_function
        self.current_function = type
        self.begin_scope()
        for param in stmt.parameters:
            self.declare(param)
            self.define(param)
        self.resolve_list(stmt.body)
        self.end_scope()
        self.current_function = enclosing_function

    def resolve_expression_stmt(self, stmt: ExpressionStmt):
        self.resolve(stmt.expression)

    def resolve_if_stmt(self, stmt: IfStmt):
        self.resolve(stmt.condition)
        self.resolve(stmt.then_branch)
        if stmt.else_branch is not None:
            self.resolve(stmt.else_branch)

    def resolve_print_stmt(self, stmt: PrintStmt):
        self.resolve(stmt.expression)

    def resolve_return_stmt(self, stmt: ReturnStmt):
        if self.current_function == FunctionType.NoFunc:
            parsing_error(stmt.keyword, "Cannot return from top-level code.")
        if stmt.value is not None:
            if self.current_function == FunctionType.Initializer:
                parsing_error(
                    stmt.keyword,
                    "Cannot return a value from an initializer."
                )
            self.resolve(stmt.value)

    def resolve_while_stmt(self, stmt: WhileStmt):
        self.resolve(stmt.condition)
        self.resolve(stmt.body)

    def resolve_binary_expr(self, expr: BinaryExpr):
        self.resolve(expr.left)
        self.resolve(expr.right)

    def resolve_call_expr(self, expr: CallExpr):
        self.resolve(expr.callee)
        for arg in expr.arguments:
            self.resolve(arg)

    def resolve_grouping_expr(self, expr: GroupingExpr):
        self.resolve(expr.expression)

    def resolve_literal_expr(self, expr: LiteralExpr):
        pass

    def resolve_logical_expr(self, expr: LogicalExpr):
        self.resolve(expr.left)
        self.resolve(expr.right)

    def resolve_unary_expr(self, expr: UnaryExpr):
        self.resolve(expr.right)

    def declare(self, name: Token):
        if len(self.scopes) == 0:
            return
        scope = self.scopes[-1]
        if name.lexeme in scope:
            parsing_error(
                name,
                "Variable with this name already declared in this scope."
            )
        scope[name.lexeme] = False

    def define(self, name: Token):
        if len(self.scopes) == 0:
            return
        self.scopes[-1][name.lexeme] = True

    def resolve_local(self, expr: Expr, name: Token):
        for i in range(len(self.scopes) - 1, -1, -1):
            if name.lexeme in self.scopes[i]:
                self.interpreter.resolve(expr, len(self.scopes) - 1 - i)
                return

    def resolve_class_stmt(self, stmt: ClassStmt):
        enclosing_class = self.current_class
        self.current_class = ClassType.Class
        self.declare(stmt.name)
        if stmt.superclass is not None:
            self.current_class = ClassType.Subclass
            self.resolve(stmt.superclass)
        self.define(stmt.name)
        if stmt.superclass is not None:
            self.begin_scope()
            self.scopes[-1]["super"] = True
        self.begin_scope()
        self.scopes[-1]["this"] = True
        for method in stmt.methods:
            declaration = FunctionType.Method
            if method.name.lexeme == "init":
                declaration = FunctionType.Initializer
            self.resolve_function(method, declaration)
        self.end_scope()
        if stmt.superclass is not None:
            self.end_scope()
        self.current_class = enclosing_class

    def resolve_get_expr(self, expr: GetExpr):
        self.resolve(expr.object)

    def resolve_set_expr(self, expr: SetExpr):
        self.resolve(expr.value)
        self.resolve(expr.object)

    def resolve_this_expr(self, expr: ThisExpr):
        if self.current_class == ClassType.NoClass:
            parsing_error(expr.keyword, "Cannot use 'this' outside of a class.")
        self.resolve_local(expr, expr.keyword)

    def resolve_super_expr(self, expr: SuperExpr):
        if self.current_class == ClassType.NoClass:
            parsing_error(
                expr.keyword,
                "Cannot use 'super' outside of a class."
            )
        elif self.current_class != ClassType.Subclass:
            parsing_error(
                expr.keyword,
                "Cannot use 'super' in a class with no superclass."
            )

        self.resolve_local(expr, expr.keyword)


# ------------------------------------------------------------------------------
# Interpreter.
# ------------------------------------------------------------------------------


class RuntimeError(Exception):
    def __init__(self, token: Token, message: str):
        self.token = token
        self.message = message


class Return(Exception):
    def __init__(self, value):
        self.value = value


class Environment:

    def __init__(self, enclosing: 'Environment' = None):
        self.values = dict()
        self.enclosing = enclosing

    def define(self, name: str, value: Any):
        self.values[name] = value

    def get(self, name: Token):
        if name.lexeme in self.values:
            return self.values[name.lexeme]
        if self.enclosing is not None:
            return self.enclosing.get(name)
        raise RuntimeError(name, f"Undefined variable '{name.lexeme}'.")

    def get_at(self, distance: int, name: str):
        return self.ancestor(distance).values[name]

    def ancestor(self, distance: int):
        environment = self
        for i in range(distance):
            environment = environment.enclosing
        return environment

    def assign(self, name: Token, value: Any):
        if name.lexeme in self.values:
            self.values[name.lexeme] = value
            return
        if self.enclosing is not None:
            self.enclosing.assign(name, value)
            return
        raise RuntimeError(name, f"Undefined variable '{name.lexeme}'.")

    def assign_at(self, distance: int, name: Token, value: Any):
        self.ancestor(distance).values[name.lexeme] = value

    def __str__(self):
        out = str(self.values)
        if self.enclosing is not None:
            out += "\n" + str(self.enclosing)
        else:
            out += "\n No Enclosing"
        return out


class Interpreter:

    def __init__(self):
        self.globals = Environment()
        self.globals.define("clock", ClockBuiltin())
        self.environment = self.globals
        self.locals = dict()

    def interpret(self, program: List[Stmt]):
        try:
            for statement in program:
                self.execute(statement)
        except RuntimeError as error:
            runtime_error(error)

    def resolve(self, expr: Expr, depth: int):
        self.locals[expr] = depth

    # ------------------------------------------------------------------------
    # Execute statements.
    # ------------------------------------------------------------------------

    def execute(self, stmt: Stmt):
        if isinstance(stmt, ExpressionStmt):
            self.exec_expression_stmt(stmt)
        elif isinstance(stmt, PrintStmt):
            self.exec_print_stmt(stmt)
        elif isinstance(stmt, VarStmt):
            self.exec_var_stmt(stmt)
        elif isinstance(stmt, BlockStmt):
            self.exec_block_stmt(stmt)
        elif isinstance(stmt, IfStmt):
            self.exec_if_stmt(stmt)
        elif isinstance(stmt, WhileStmt):
            self.exec_while_stmt(stmt)
        elif isinstance(stmt, FunctionStmt):
            self.exec_function_stmt(stmt)
        elif isinstance(stmt, ReturnStmt):
            self.exec_return_stmt(stmt)
        elif isinstance(stmt, ClassStmt):
            self.exec_class_stmt(stmt)

    def exec_expression_stmt(self, stmt: ExpressionStmt):
        self.eval(stmt.expression)

    def exec_print_stmt(self, stmt: PrintStmt):
        value = self.eval(stmt.expression)
        print(self.stringify(value))

    def exec_var_stmt(self, stmt: VarStmt):
        value = None
        if stmt.initializer is not None:
            value = self.eval(stmt.initializer)
        self.environment.define(stmt.name.lexeme, value)

    def exec_block_stmt(self, stmt: BlockStmt):
        self.exec_block(stmt.statements, Environment(self.environment))

    def exec_block(self, statements: List[Stmt], environment: Environment):
        previous_environment = self.environment
        try:
            self.environment = environment
            for statement in statements:
                self.execute(statement)
        finally:
            self.environment = previous_environment

    def exec_if_stmt(self, stmt: IfStmt):
        if self.is_truthy(self.eval(stmt.condition)):
            self.execute(stmt.then_branch)
        elif stmt.else_branch is not None:
            self.execute(stmt.else_branch)

    def exec_while_stmt(self, stmt: WhileStmt):
        while self.is_truthy(self.eval(stmt.condition)):
            self.execute(stmt.body)

    def exec_function_stmt(self, stmt: FunctionStmt):
        function = LoxFunction(stmt, self.environment, False)
        self.environment.define(stmt.name.lexeme, function)

    def exec_return_stmt(self, stmt: ReturnStmt):
        value = None
        if stmt.value is not None:
            value = self.eval(stmt.value)
        raise Return(value)

    def exec_class_stmt(self, stmt: ClassStmt):
        superclass = None
        if stmt.superclass is not None:
            superclass = self.eval(stmt.superclass)
            if not isinstance(superclass, LoxClass):
                raise RuntimeError(
                    stmt.superclass.name,
                    "Superclass must be a class."
                )
        self.environment.define(stmt.name.lexeme, None)
        if stmt.superclass is not None:
            self.environment = Environment(self.environment)
            self.environment.define("super", superclass)
        methods = dict()
        for method in stmt.methods:
            is_initializer = method.name.lexeme == "init"
            function = LoxFunction(method, self.environment, is_initializer)
            methods[method.name.lexeme] = function
        klass = LoxClass(stmt.name.lexeme, superclass, methods)
        if superclass is not None:
            self.environment = self.environment.enclosing
        self.environment.assign(stmt.name, klass)

    # ------------------------------------------------------------------------
    # Evaluate expressions.
    # ------------------------------------------------------------------------

    def eval(self, expr: Expr):
        if isinstance(expr, LiteralExpr):
            return self.eval_literal(expr)
        elif isinstance(expr, GroupingExpr):
            return self.eval_grouping(expr)
        elif isinstance(expr, UnaryExpr):
            return self.eval_unary(expr)
        elif isinstance(expr, BinaryExpr):
            return self.eval_binary(expr)
        elif isinstance(expr, VariableExpr):
            return self.eval_variable(expr)
        elif isinstance(expr, AssignExpr):
            return self.eval_assign(expr)
        elif isinstance(expr, LogicalExpr):
            return self.eval_logical(expr)
        elif isinstance(expr, CallExpr):
            return self.eval_call(expr)
        elif isinstance(expr, GetExpr):
            return self.eval_get(expr)
        elif isinstance(expr, SetExpr):
            return self.eval_set(expr)
        elif isinstance(expr, ThisExpr):
            return self.lookup_variable(expr.keyword, expr)
        elif isinstance(expr, SuperExpr):
            return self.eval_super(expr)

    def eval_literal(self, expr: LiteralExpr):
        return expr.value

    def eval_grouping(self, expr: GroupingExpr):
        return self.eval(expr.expression)

    def eval_unary(self, expr: UnaryExpr):
        right = self.eval(expr.right)
        if expr.operator.type == TokType.Minus:
            self.check_float_operand(expr.operator, right)
            return -right
        elif expr.operator.type == TokType.Bang:
            return not self.is_truthy(right)

    def eval_binary(self, expr: BinaryExpr):
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

    def eval_variable(self, expr: VariableExpr):
        return self.lookup_variable(expr.name, expr)

    def eval_assign(self, expr: AssignExpr):
        value = self.eval(expr.value)
        if expr in self.locals:
            distance = self.locals[expr]
            self.environment.assign_at(distance, expr.name, value)
        else:
            self.globals.assign(expr.name, value)
        return value

    def eval_logical(self, expr: LogicalExpr):
        left = self.eval(expr.left)
        if expr.operator.type == TokType.Or:
            if self.is_truthy(left):
                return left
        else:
            if not self.is_truthy(left):
                return left
        return self.eval(expr.right)

    def eval_call(self, expr: CallExpr):
        callee = self.eval(expr.callee)
        arguments = list()
        for argument in expr.arguments:
            arguments.append(self.eval(argument))
        if not hasattr(callee, 'call'):
            raise RuntimeError(
                expr.paren,
                "Can only call functions and classes.")
        if len(arguments) != callee.arity():
            raise RuntimeError(
                expr.paren,
                f"Expected {callee.arity()} arguments, got {len(arguments)}.")
        return callee.call(self, arguments)

    def eval_get(self, expr: GetExpr):
        object = self.eval(expr.object)
        if isinstance(object, LoxInstance):
            return object.get(expr.name)
        raise RuntimeError(expr.name, "Only instances have properties.")

    def eval_set(self, expr: SetExpr):
        object = self.eval(expr.object)
        if not isinstance(object, LoxInstance):
            raise RuntimeError(expr.name, "Only instances have fields.")
        value = self.eval(expr.value)
        object.set(expr.name, value)
        return value

    def eval_super(self, expr: SuperExpr):
        distance = self.locals.get(expr)
        superclass = self.environment.get_at(distance, "super")
        object = self.environment.get_at(distance - 1, "this")
        method = superclass.find_method(object, expr.method.lexeme)
        if method is None:
            raise RuntimeError(
                expr.method,
                f"Undefined property '{expr.method.lexeme}'."
            )
        return method


    # ------------------------------------------------------------------------
    # Helpers.
    # ------------------------------------------------------------------------

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

    def lookup_variable(self, name: Token, expr: Expr):
        if expr in self.locals:
            distance = self.locals[expr]
            return self.environment.get_at(distance, name.lexeme)
        return self.globals.get(name)


class ClockBuiltin:

    def call(self, interpreter: 'Interpreter', arguments: List):
        return time.perf_counter()

    def arity(self):
        return 0

    def __str__(self):
        return "<builtin fn: clock>"


class LoxFunction:

    def __init__(self, declaration: FunctionStmt, closure: Environment, is_initializer: bool):
        self.declaration = declaration
        self.closure = closure
        self.is_initializer = is_initializer

    def call(self, interpreter: Interpreter, arguments: List):
        environment = Environment(self.closure)
        for i in range(len(self.declaration.parameters)):
            environment.define(
                self.declaration.parameters[i].lexeme,
                arguments[i]
            )
        try:
            interpreter.exec_block(self.declaration.body, environment)
        except Return as return_value:
            if self.is_initializer:
                return self.closure.get_at(0, "this")
            return return_value.value
        if self.is_initializer:
            return self.closure.get_at(0, "this")
        return None

    def arity(self):
        return len(self.declaration.parameters)

    def __str__(self):
        return f"<fn {declaration.name.lexeme}>"

    def bind(self, instance: 'LoxInstance'):
        environment = Environment(self.closure)
        environment.define("this", instance)
        return LoxFunction(self.declaration, environment, self.is_initializer)


class LoxClass:

    def __init__(self, name: str, superclass: 'LoxClass', methods: Dict[str, LoxFunction]):
        self.name = name
        self.methods = methods
        self.superclass = superclass

    def __str__(self):
        return self.name

    def call(self, interpreter: Interpreter, arguments: List):
        instance = LoxInstance(self)
        if "init" in self.methods:
            initializer = self.methods["init"]
            initializer.bind(instance).call(interpreter, arguments)
        return instance

    def arity(self):
        if "init" in self.methods:
            return self.methods["init"].arity()
        return 0

    def find_method(self, instance: 'LoxInstance', name: str):
        if name in self.methods:
            return self.methods[name].bind(instance)
        if self.superclass is not None:
            return self.superclass.find_method(instance, name)
        return None


class LoxInstance:

    def __init__(self, klass: LoxClass):
        self.klass = klass
        self.fields = dict()

    def __str__(self):
        return f"{self.klass.name} instance"

    def get(self, name: Token):
        if name.lexeme in self.fields:
            return self.fields[name.lexeme]
        method = self.klass.find_method(self, name.lexeme)
        if method is  not None:
            return method
        raise RuntimeError(name, f"Undefined property '{name.lexeme}'.")

    def set(self, name: Token, value: Any):
        self.fields[name.lexeme] = value


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
    if had_lexing_error:
        return

    parser = Parser(tokens)
    statements = parser.parse()
    if had_parsing_error:
        return

    resolver = Resolver(interpreter)
    resolver.resolve_program(statements)
    if had_parsing_error:
        return

    interpreter.interpret(statements)


def run_prompt():
    try:
        while True:
            print("> ", end="")
            run(input())
    except KeyboardInterrupt:
        print()


def run_file(path):
    run(open(path).read())
    if had_lexing_error:
        sys.exit(1)
    if had_parsing_error:
        sys.exit(2)
    if had_runtime_error:
        sys.exit(3)


def main():
    if len(sys.argv) > 2:
        print("Usage: lox [script]")
    elif len(sys.argv) == 2:
        run_file(sys.argv[1])
    else:
        run_prompt()


if __name__ == '__main__':
    main()
