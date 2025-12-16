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

export default function CpuRamBarChart({ data }: { data: any[] }) {
  return (
    <ChartWrapper
      title="CPU vs RAM Usage"
      description="Resource utilization comparison across models"
    >
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data}>
          <XAxis dataKey="model" hide />
          <YAxis />
          <Tooltip />
          <Bar dataKey="cpu_percent" />
          <Bar dataKey="ram_mb" />
        </BarChart>
      </ResponsiveContainer>
    </ChartWrapper>
  )
}
