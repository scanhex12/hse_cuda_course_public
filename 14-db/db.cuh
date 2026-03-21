#pragma once

#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

enum class GpuOp : int { Add = 0, Sub = 1, Mul = 2, Div = 3 };

__global__ void BinaryKernel(const int *a, const int *b, int *out, int n,
                             GpuOp op);

enum class TokenType {
    Identifier,
    Dot,
    Plus,
    Minus,
    Mul,
    Div,
    LParen,
    RParen,
    End
};

struct Token {
    TokenType type;
    std::string text;
};

class Lexer {
  public:
    explicit Lexer(const std::string &source);
    Token Next();

  private:
    Token ParseIdentifier();
    void SkipWhitespace();

  private:
    const std::string &src_;
    size_t pos_ = 0;
};

struct AstNode {
    virtual ~AstNode() = default;
};

struct ColumnRef : public AstNode
{
    ColumnRef(std::string table, std::string column);

    std::string table;
    std::string column;
};

struct BinaryExpr : public AstNode {
    BinaryExpr(char op, std::shared_ptr<AstNode> lhs,
               std::shared_ptr<AstNode> rhs);

    char op;
    std::shared_ptr<AstNode> lhs;
    std::shared_ptr<AstNode> rhs;
};

class Parser {
  public:
    explicit Parser(Lexer &lexer);
    std::shared_ptr<AstNode> Parse();

  private:
    std::shared_ptr<AstNode> ParseExpression();
    std::shared_ptr<AstNode> ParseTerm();
    std::shared_ptr<AstNode> ParseFactor();

    void Expect(TokenType type);
    void Advance();

  private:
    Lexer &lexer_;
    Token current_;
};

enum class OpType { Load, Add, Sub, Mul, Div };

struct ComputeNode {
    ComputeNode(OpType op, int lhs, int rhs);
    ComputeNode(OpType op, int lhs, int rhs, std::string table,
                std::string column);

    OpType op;
    int lhs = -1;
    int rhs = -1;
    std::string table;
    std::string column;
};

class Database {
  public:
    ~Database();

    void AddTable(const std::string &table,
                  const std::vector<std::string> &columns,
                  const std::vector<std::vector<int>> &data);

    std::vector<int> Execute(const std::string &query);

  private:
    /// Your functions here

    std::unordered_map<std::string, std::vector<std::string>> schema_;
    std::unordered_map<std::string, std::vector<std::vector<int>>> hostData_;

    std::unordered_map<std::string, std::vector<int *>> deviceColumns_;
    std::unordered_map<std::string, size_t> tableSizes_;

    std::vector<ComputeNode> nodes_;
    std::vector<int *> tempBuffers_;
    std::unordered_map<int, int *> graphOutputs_;

    cudaGraph_t graph_{};
    cudaGraphExec_t graphExec_{};
};
