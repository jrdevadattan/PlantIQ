// Shared mock data for the manufacturing intelligence dashboard

export interface BatchRecord {
  id: string;
  timestamp: string;
  temperature: number;
  conveyorSpeed: number;
  holdTime: number;
  batchSize: number;
  materialType: number;
  hourOfDay: number;
  qualityScore: number;
  yieldPct: number;
  performancePct: number;
  energyKwh: number;
  status: "completed" | "running" | "scheduled" | "alert";
  anomalyScore: number;
}

export interface PredictionResult {
  qualityScore: number;
  yieldPct: number;
  performancePct: number;
  energyKwh: number;
  confidence: {
    quality: [number, number];
    yield: [number, number];
    performance: [number, number];
    energy: [number, number];
  };
}

export interface ShapValue {
  feature: string;
  contribution: number;
  direction: "positive" | "negative";
}

export interface AnomalyResult {
  score: number;
  isAnomaly: boolean;
  threshold: number;
  diagnosis: string;
  diagnosisConfidence: number;
  humanReadable: string;
  recommendedAction: string;
  estimatedImpact: number;
}

// Recent batch data
export const recentBatches: BatchRecord[] = [
  {
    id: "B-2026-0301-001",
    timestamp: "2026-03-01 08:12:33",
    temperature: 183,
    conveyorSpeed: 76,
    holdTime: 18,
    batchSize: 500,
    materialType: 1,
    hourOfDay: 8,
    qualityScore: 91.2,
    yieldPct: 93.4,
    performancePct: 92.1,
    energyKwh: 38.8,
    status: "completed",
    anomalyScore: 0.08,
  },
  {
    id: "B-2026-0301-002",
    timestamp: "2026-03-01 09:45:10",
    temperature: 178,
    conveyorSpeed: 72,
    holdTime: 22,
    batchSize: 480,
    materialType: 2,
    hourOfDay: 9,
    qualityScore: 88.5,
    yieldPct: 90.1,
    performancePct: 89.7,
    energyKwh: 44.2,
    status: "completed",
    anomalyScore: 0.71,
  },
  {
    id: "B-2026-0301-003",
    timestamp: "2026-03-01 11:23:05",
    temperature: 185,
    conveyorSpeed: 78,
    holdTime: 15,
    batchSize: 520,
    materialType: 1,
    hourOfDay: 11,
    qualityScore: 93.8,
    yieldPct: 95.2,
    performancePct: 94.3,
    energyKwh: 36.1,
    status: "completed",
    anomalyScore: 0.05,
  },
  {
    id: "B-2026-0302-001",
    timestamp: "2026-03-02 07:30:00",
    temperature: 181,
    conveyorSpeed: 74,
    holdTime: 20,
    batchSize: 510,
    materialType: 1,
    hourOfDay: 7,
    qualityScore: 90.4,
    yieldPct: 92.8,
    performancePct: 93.5,
    energyKwh: 40.1,
    status: "running",
    anomalyScore: 0.12,
  },
  {
    id: "B-2026-0302-002",
    timestamp: "2026-03-02 09:00:00",
    temperature: 186,
    conveyorSpeed: 80,
    holdTime: 16,
    batchSize: 490,
    materialType: 2,
    hourOfDay: 9,
    qualityScore: 0,
    yieldPct: 0,
    performancePct: 0,
    energyKwh: 0,
    status: "scheduled",
    anomalyScore: 0,
  },
  {
    id: "B-2026-0228-008",
    timestamp: "2026-02-28 16:10:22",
    temperature: 190,
    conveyorSpeed: 82,
    holdTime: 25,
    batchSize: 550,
    materialType: 2,
    hourOfDay: 16,
    qualityScore: 85.3,
    yieldPct: 87.6,
    performancePct: 86.2,
    energyKwh: 52.4,
    status: "alert",
    anomalyScore: 0.82,
  },
  {
    id: "B-2026-0228-007",
    timestamp: "2026-02-28 14:05:18",
    temperature: 182,
    conveyorSpeed: 75,
    holdTime: 17,
    batchSize: 500,
    materialType: 1,
    hourOfDay: 14,
    qualityScore: 92.1,
    yieldPct: 94.0,
    performancePct: 91.8,
    energyKwh: 37.5,
    status: "completed",
    anomalyScore: 0.06,
  },
  {
    id: "B-2026-0228-006",
    timestamp: "2026-02-28 12:30:45",
    temperature: 179,
    conveyorSpeed: 71,
    holdTime: 19,
    batchSize: 460,
    materialType: 1,
    hourOfDay: 12,
    qualityScore: 89.8,
    yieldPct: 91.5,
    performancePct: 90.4,
    energyKwh: 39.7,
    status: "completed",
    anomalyScore: 0.15,
  },
];

// Live power curve data (simulated 1Hz readings for ~8 minutes so far)
export const generatePowerCurve = (type: "normal" | "bearing_wear" | "wet_material" = "normal", points = 480): number[] => {
  const data: number[] = [];
  for (let i = 0; i < points; i++) {
    const t = i;
    let base = 3 + 4 * (1 - Math.exp(-t / 120)) - 2 * (1 - Math.exp(-(1800 - t) / 120));

    if (type === "bearing_wear") {
      base += 0.003 * t;
    } else if (type === "wet_material") {
      base += 0.8 * Math.sin(t / 30) * (0.5 + Math.random());
    } else {
      base += (Math.random() - 0.5) * 0.2;
    }

    data.push(Math.max(0, Number(base.toFixed(2))));
  }
  return data;
};

// SHAP values mock for energy prediction
export const mockShapValues: ShapValue[] = [
  { feature: "Hold Time", contribution: 3.8, direction: "positive" },
  { feature: "Material Type", contribution: 1.9, direction: "positive" },
  { feature: "Conveyor Speed", contribution: 1.1, direction: "positive" },
  { feature: "Batch Size", contribution: 0.4, direction: "positive" },
  { feature: "Hour of Day", contribution: -0.1, direction: "negative" },
  { feature: "Temperature", contribution: -0.3, direction: "negative" },
];

// Sliding window prediction timeline
export const slidingWindowData = [
  { time: "0:00", energy: 38.8, lower: 32.8, upper: 44.8, confidence: 65 },
  { time: "2:00", energy: 39.2, lower: 34.2, upper: 44.2, confidence: 72 },
  { time: "4:00", energy: 40.1, lower: 36.1, upper: 44.1, confidence: 78 },
  { time: "6:00", energy: 42.3, lower: 39.3, upper: 45.3, confidence: 84 },
  { time: "8:00", energy: 44.5, lower: 42.0, upper: 47.0, confidence: 89 },
  { time: "10:00", energy: 43.8, lower: 42.0, upper: 45.6, confidence: 91 },
  { time: "12:00", energy: 43.2, lower: 41.8, upper: 44.6, confidence: 93 },
  { time: "14:00", energy: 42.8, lower: 41.6, upper: 44.0, confidence: 95 },
  { time: "16:00", energy: 42.5, lower: 41.5, upper: 43.5, confidence: 96 },
];

// Energy trend over past 30 batches
export const energyTrendData = Array.from({ length: 30 }, (_, i) => ({
  batch: i + 1,
  actual: 35 + Math.random() * 15,
  predicted: 36 + Math.random() * 12,
  target: 40,
})).map(d => ({
  ...d,
  actual: Number(d.actual.toFixed(1)),
  predicted: Number(d.predicted.toFixed(1)),
}));

// Performance distribution data
export const performanceDistribution = [
  { range: "60-70%", count: 2 },
  { range: "70-80%", count: 5 },
  { range: "80-85%", count: 12 },
  { range: "85-90%", count: 35 },
  { range: "90-95%", count: 68 },
  { range: "95-100%", count: 28 },
];

// Shift performance data
export const shiftPerformance = [
  { shift: "Morning (6-14)", quality: 92.4, yield: 94.1, energy: 38.2, batches: 45 },
  { shift: "Afternoon (14-22)", quality: 89.8, yield: 91.3, energy: 41.5, batches: 42 },
  { shift: "Night (22-6)", quality: 87.2, yield: 88.9, energy: 43.1, batches: 38 },
];

// Daily energy consumption
export const dailyEnergy = [
  { day: "Mon", kwh: 385 },
  { day: "Tue", kwh: 412 },
  { day: "Wed", kwh: 398 },
  { day: "Thu", kwh: 445 },
  { day: "Fri", kwh: 378 },
  { day: "Sat", kwh: 290 },
  { day: "Sun", kwh: 265 },
];
