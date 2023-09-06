## Introduction
![[Pasted image 20230815103529.png]]
(Cueva and Wei, 2018)

---
## First model
- RNN, 100 cells
- Input: 1D speeds
- Desired output (decoder): 1D positions
- Random positive speeds

---
### Results
![[Pasted image 20230316130611.png]]
- Accurate and robust, but not energy efficient

---
## Second model: Mini-model
- RNN, 2 cells
- Input: 1D speeds
- Desired output (without decoder): 1D positions mapped on 2D unit circle
- Random positive speeds

---
### Results
![[Pasted image 20230323133935.png]]
- Rotation matrix: energy efficient, but neither accurate nor robust

---
## Third model: With regularization
- RNN, 100 cells
- Input: 1D speeds
- Desired output (decoder): 1D positions
- Random positive speeds
- *L2 regularization* on the vector norm of cell activity

---
### Results
![[Pasted image 20230316164401.png]]
- Robust and accurate, but not energy efficient
---
## Fourth model: Arccos-loss
- RNN, 100 cells
- Input: 1D speeds
- Output: *relative* 1D positions decoded from relative 100D vector angles with arccos
- Random positive speeds

---
### Results

![[Pasted image 20230620132555.png]]
- 93 % explained variance

---
![[Pasted image 20230620132910.png]]
- x-axis: position (scaled down from 0 to 2pi, to 0 to 1), y-axis: cell activity

---
![[Pasted image 20230620132449.png]]

---
![[Pasted image 20230620132705.png]]

---
![[Pasted image 20230620132730.png]]

---
- In other words: robust, accurate and energy efficient

---
# Navigation in 2D

---
![[Pasted image 20230809102204.png]]

---
## Comparison model
- RNN, 288 cells
- Input: 2D speeds
- Output (decoder): 2D positions
- No regularization

---
### Result
![[Pasted image 20230818142503.png]]

---
![[Pasted image 20230818142747.png]]
- 87 % explained variance

---
- In other words: good accuracy and robustness, but bad energy efficiency

---
## Trivial expansion
- Two instances of 1D arccos model, one for x and one for y axis

---
### Results
![[Pasted image 20230622141915.png]]

---
![[Pasted image 20230622140803.png]]

---
### Weaknesses
- Doesn't make sense with hexagonal grid cells
- Only one frequency
- Restrictive

---
## Low et al.
- RNN, 256 cells
- Input: 2D speeds
- Output (decoder): sin and cos of x and y positions (4 values)

---
### Results
![[Pasted image 20230809125658.png]]

---
![[Pasted image 20230809125754.png]]

---
![[Pasted image 20230807132838.png]]

---
![[Pasted image 20230807132853.png]]

---
![[Pasted image 20230808112822.png]]

---
![[Pasted image 20230808112900.png]]

---
### Comments
- Energy efficient, robust and accurate
- One frequency
- Worse explainability with trainable decoder

---
## Takeaways
