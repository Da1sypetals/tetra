// main.swift - Examples of using the Tensor implementation
// Created on 2025-11-15

import Foundation

// Create a leaf node (tensor with no operations)
let leafTensor = Node.leaf(
    shape: [
        .Static(2),
        .Dynamic("batch_size"),
    ], dtype: .float32)
print("Leaf tensor: \(leafTensor.description())")
print("Number of dimensions: \(leafTensor.ndim())")
print()

// Example with higher dimensional tensor
let highDimTensor = Node.leaf(shape: [ShapeType.Static(2), ShapeType.Static(3)], dtype: .float16)
print("High dimensional tensor: \(highDimTensor.description())")
print("Number of dimensions: \(highDimTensor.ndim())")
print()

// Example with different data types using UnaryOp
let sigmoidOpFloat32 = UnaryOp(name: "sigmoid", inputDtype: .float32, outputDtype: .float32)
let sigmoidOpFloat16 = UnaryOp(name: "sigmoid", inputDtype: .float16, outputDtype: .float16)
let res1 = sigmoidOpFloat32.call(leafTensor)
let res2 = sigmoidOpFloat16.call(highDimTensor)
print(res1.description())
print(res2.description())
print()

// Test broadcast function
print("=== Testing Broadcast Function ===")

// Test case 1: Simple compatible shapes
let shape1 = [ShapeType.Static(2), ShapeType.Static(3)]
let shape2 = [ShapeType.Static(3)]
if let broadcasted1 = broadcast(shape1, shape2) {
    print("Broadcasted shape1: \(broadcasted1)")
} else {
    print("Shapes are not broadcastable")
}

// Test case 2: Shapes with dimension 1
let shape3 = [ShapeType.Static(1), ShapeType.Static(4), ShapeType.Static(1)]
let shape4 = [ShapeType.Static(3), ShapeType.Static(1), ShapeType.Static(5)]
if let broadcasted2 = broadcast(shape3, shape4) {
    print("Broadcasted shape2: \(broadcasted2)")
} else {
    print("Shapes are not broadcastable")
}

// Test case 3: Incompatible shapes
let shape5 = [ShapeType.Static(2), ShapeType.Static(3)]
let shape6 = [ShapeType.Static(4), ShapeType.Static(5)]
if let broadcasted3 = broadcast(shape5, shape6) {
    print("Broadcasted shape3: \(broadcasted3)")
} else {
    print("Shapes are not broadcastable")
}

// Test case 4: Multiple shapes
let shape7 = [ShapeType.Static(2), ShapeType.Static(1)]
let shape8 = [ShapeType.Static(1), ShapeType.Static(3)]
let shape9 = [ShapeType.Static(3)]
if let broadcasted4 = broadcast(shape7, shape8, shape9) {
    print("Broadcasted shape4: \(broadcasted4)")
} else {
    print("Shapes are not broadcastable")
}

// Test case 5: Dynamic shapes
let shape10 = [ShapeType.Dynamic("batch"), ShapeType.Static(1)]
let shape11 = [ShapeType.Static(1), ShapeType.Static(3)]
if let broadcasted5 = broadcast(shape10, shape11) {
    print("Broadcasted shape5: \(broadcasted5)")
} else {
    print("Shapes are not broadcastable")
}

// Test case 6: Dynamic shapes with different names (should not be broadcastable)
let shape12 = [ShapeType.Dynamic("batch"), ShapeType.Static(1)]
let shape13 = [ShapeType.Dynamic("sequence"), ShapeType.Static(3)]
if let broadcasted6 = broadcast(shape12, shape13) {
    print("Broadcasted shape6: \(broadcasted6)")
} else {
    print("Shapes are not broadcastable")
}

// Test case 7: Dynamic shapes with same name (should be broadcastable)
let shape14 = [ShapeType.Dynamic("batch"), ShapeType.Static(1)]
let shape15 = [ShapeType.Dynamic("batch"), ShapeType.Static(3)]
if let broadcasted7 = broadcast(shape14, shape15) {
    print("Broadcasted shape7: \(broadcasted7)")
} else {
    print("Shapes are not broadcastable")
}
print()

// Test BinaryOp with a success example
print("=== Testing BinaryOp ===")

// Create two tensors with compatible shapes
let tensor1 = Node.leaf(shape: [ShapeType.Static(2), ShapeType.Static(3)], dtype: .float32)
let tensor2 = Node.leaf(shape: [ShapeType.Static(3)], dtype: .float32)

// Create a binary operation (addition)
let addOp = BinaryOp(name: "add", leftDtype: .float32, rightDtype: .float32, outputDtype: .float32)

// Apply the operation
let result = addOp.call(tensor1, tensor2)

// Print the results
print("Tensor1: \(tensor1.description())")
print("Tensor2: \(tensor2.description())")
print("Result: \(result.description())")
print()

// Test with different shapes that can be broadcasted
let tensor3 = Node.leaf(
    shape: [ShapeType.Static(1), ShapeType.Static(4), ShapeType.Static(1)], dtype: .float16)
let tensor4 = Node.leaf(
    shape: [ShapeType.Dynamic("first"), ShapeType.Static(1), ShapeType.Static(5)], dtype: .float16)

// Create a multiplication operation
let mulOp = BinaryOp(name: "mul", leftDtype: .float16, rightDtype: .float16, outputDtype: .float16)

// Apply the operation
let result2 = mulOp.call(tensor3, tensor4)

// Print the results
print("Tensor3: \(tensor3.description())")
print("Tensor4: \(tensor4.description())")
print("Result2: \(result2.description())")
print()
