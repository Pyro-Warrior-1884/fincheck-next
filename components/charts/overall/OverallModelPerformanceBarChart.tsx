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

export default function OverallModelPerformanceBarChart({
  data,
}: {
  data: any[]
}) {
  return (
    <ChartWrapper
      title="Overall Model Performance Index"
      description="Composite score balancing accuracy, latency, and memory"
    >
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data}>
          <XAxis dataKey="model" hide />
          <YAxis />
          <Tooltip />
          <Bar dataKey="score" />
        </BarChart>
      </ResponsiveContainer>
    </ChartWrapper>
  )
}
