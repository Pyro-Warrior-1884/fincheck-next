"use client"

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts"
import ChartWrapper from "../ChartWrapper"

export default function EntropyConfidenceBarChart({
  data,
}: {
  data: any[]
}) {
  return (
    <ChartWrapper
      title="Prediction Entropy vs Confidence"
      description="Lower entropy indicates higher prediction certainty"
    >
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data}>
          <XAxis dataKey="model" hide />
          <YAxis />
          <Tooltip />
          <Bar dataKey="entropy" />
          <Bar dataKey="confidence" />
        </BarChart>
      </ResponsiveContainer>
    </ChartWrapper>
  )
}
