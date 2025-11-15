// main.swift - Examples of using the Tensor implementation
// Created on 2025-11-15

import Foundation

// Create a leaf node (tensor with no operations)
let leafTensor = Node.leaf(shape: [ShapeType.Static(2), ShapeType.Static(3)], dtype: .float32)
print("Leaf tensor: \(leafTensor.description())")
print("Number of dimensions: \(leafTensor.ndim())")
print()

// Create a unary operation node
let unaryTensor = Node.unary(
    name: "relu",
    shape: [ShapeType.Static(2), ShapeType.Dynamic("batch_size")],
    dtype: .float32,
    operand: leafTensor
)
print("Unary tensor (ReLU): \(unaryTensor.description())")
print("Number of dimensions: \(unaryTensor.ndim())")
print()

// Create another leaf node for binary operation
let anotherLeafTensor = Node.leaf(
    shape: [ShapeType.Static(2), ShapeType.Static(3)], dtype: .float32)

// Create a binary operation node
let binaryTensor = Node.binary(
    name: "add",
    shape: [ShapeType.Static(2), ShapeType.Static(3)],
    dtype: .float32,
    left: leafTensor,
    right: anotherLeafTensor
)
print("Binary tensor (Add): \(binaryTensor.description())")
print("Number of dimensions: \(binaryTensor.ndim())")
print()

// Example with different data types
let intTensor = Node.leaf(shape: [ShapeType.Static(2), ShapeType.Static(3)], dtype: .int32)
print("Int tensor: \(intTensor.description())")
print("Number of dimensions: \(intTensor.ndim())")
print()

// Example with higher dimensional tensor
let highDimTensor = Node.leaf(shape: [ShapeType.Static(2), ShapeType.Static(3)], dtype: .float16)
print("High dimensional tensor: \(highDimTensor.description())")
print("Number of dimensions: \(highDimTensor.ndim())")
print()

// Example of nested operations
let nestedTensor = Node.unary(
    name: "sigmoid",
    shape: [ShapeType.Static(2), ShapeType.Static(3)],
    dtype: .float32,
    operand: binaryTensor
)
print("Nested operation tensor (Sigmoid after Add): \(nestedTensor.description())")
print("Number of dimensions: \(nestedTensor.ndim())")
print()

// Examples using UnaryOp class
print("=== Examples using UnaryOp class ===")

// Create a UnaryOp instance for ReLU
let reluOp = UnaryOp(name: "relu")
let reluResult = reluOp.call(leafTensor)
print("ReLU operation result: \(reluResult.description())")

// Create a UnaryOp instance for Sigmoid
let sigmoidOp = UnaryOp(name: "sigmoid")
let sigmoidResult = sigmoidOp.call(leafTensor)
print("Sigmoid operation result: \(sigmoidResult.description())")

// Chain operations using UnaryOp
let tanhOp = UnaryOp(name: "tanh")
let tanhResult = tanhOp.call(sigmoidResult)
print("Tanh after Sigmoid result: \(tanhResult.description())")

// Apply UnaryOp to binary operation result
let reluOnBinary = reluOp.call(binaryTensor)
print("ReLU on binary operation result: \(reluOnBinary.description())")

// Example with different data types
let reluOnInt = reluOp.call(intTensor)
print("ReLU on int tensor result: \(reluOnInt.description())")
