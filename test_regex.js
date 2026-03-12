const text1 = "bisect(*[int(";
console.log(text1.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
const text2 = "\\\\"; // single backslash
console.log(text2.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
